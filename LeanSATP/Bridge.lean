import Aesop
import Lean

open Lean Parser Elab Tactic Meta
open System (FilePath)

namespace LeanSATP

private def defaultServerHost : String := "127.0.0.1"
private def defaultServerPort : Nat := 5177
private def defaultRequestTimeout : Nat := 120

private def envNat? (name : String) : IO (Option Nat) := do
  match ← IO.getEnv name with
  | some s => pure s.trim.toNat?
  | none => pure none

private def hasRuntimeFiles (path : FilePath) : IO Bool := do
  let pyproject := path / "pyproject.toml"
  let runtime := path / "python" / "src" / "leansatp_runtime" / "service.py"
  return (← pyproject.pathExists) && (← runtime.pathExists)

private partial def findPackageRootFrom (path : FilePath) (fuel : Nat := 16) : IO FilePath := do
  if fuel == 0 then
    return path.normalize
  if ← hasRuntimeFiles path then
    return path.normalize
  match path.parent with
  | some parent => findPackageRootFrom parent (fuel - 1)
  | none => return path.normalize

private def packageRoot : IO FilePath := do
  let oleanPath ← Lean.findOLean `LeanSATP.Bridge
  let start := oleanPath.parent.getD oleanPath
  let root ← findPackageRootFrom start
  if ← hasRuntimeFiles root then
    Lean.realPathNormalized root
  else
    throw <| IO.userError s!"LeanSATP package root not found from {oleanPath}"

private def defaultRepoRoot : IO FilePath := do
  Lean.realPathNormalized (← packageRoot)

private def defaultCacheDir : IO String := do
  return (((← defaultRepoRoot) / "cache").normalize.toString)

private def defaultCheckpointFromCacheDir (cacheDir : String) : String :=
  ((FilePath.mk cacheDir) / "best_checkpoint.pt").normalize.toString

structure RuntimeConfig where
  repoRoot : FilePath
  cacheDir : String
  checkpoint : String
  host : String
  port : Nat
  requestTimeout : Nat

structure ServiceRunner where
  probe : String
  cmd : String
  args : Array String

private def runtimeConfigFromEnv : IO RuntimeConfig := do
  let repoRoot ← defaultRepoRoot
  let cacheDirDefault ← defaultCacheDir
  let checkpointEnv? ← IO.getEnv "SATP_CHECKPOINT"
  let cacheDir ←
    match (← IO.getEnv "SATP_CACHE_DIR"), checkpointEnv? with
    | some value, _ => pure value
    | none, some checkpoint =>
        let checkpointPath := FilePath.mk checkpoint
        pure ((checkpointPath.parent.getD (FilePath.mk cacheDirDefault)).normalize.toString)
    | none, none => pure cacheDirDefault
  let host := ((← IO.getEnv "SATP_SERVER_HOST").getD defaultServerHost).trim
  let port := (← envNat? "SATP_SERVER_PORT").getD defaultServerPort
  let requestTimeout := (← envNat? "SATP_REQUEST_TIMEOUT").getD defaultRequestTimeout
  return {
    repoRoot := repoRoot
    cacheDir := cacheDir
    checkpoint := checkpointEnv?.getD (defaultCheckpointFromCacheDir cacheDir)
    host := if host.isEmpty then defaultServerHost else host
    port := port
    requestTimeout := requestTimeout
  }

private def serviceModuleArgs (cfg : RuntimeConfig) (mode : String) : Array String :=
  let args := #[
    "-m",
    "leansatp_runtime.service",
    mode,
    "--checkpoint",
    cfg.checkpoint,
    "--cache-dir",
    cfg.cacheDir
  ]
  if mode = "--serve" then
    args ++ #[
      "--host",
      cfg.host,
      "--port",
      toString cfg.port
    ]
  else
    args

private def serviceRunners (cfg : RuntimeConfig) (mode : String) : Array ServiceRunner :=
  let moduleArgs := serviceModuleArgs cfg mode
  #[
    {
      probe := "uv"
      cmd := "uv"
      args := #["run", "python"] ++ moduleArgs
    },
    {
      probe := "python3"
      cmd := "env"
      args := #["PYTHONPATH=python/src", "python3"] ++ moduleArgs
    },
    {
      probe := "python"
      cmd := "env"
      args := #["PYTHONPATH=python/src", "python"] ++ moduleArgs
    }
  ]

private def describeRunner (runner : ServiceRunner) : String :=
  match runner.probe with
  | "uv" => "uv run python -m leansatp_runtime.service"
  | "python3" => "PYTHONPATH=python/src python3 -m leansatp_runtime.service"
  | "python" => "PYTHONPATH=python/src python -m leansatp_runtime.service"
  | other => other

private def pythonEnvironmentMessage (cfg : RuntimeConfig) : MessageData :=
  m!"satp: no Python runtime found under {cfg.repoRoot}\n" ++
  m!"  • Install uv (recommended): curl -LsSf https://astral.sh/uv/install.sh | sh\n" ++
  m!"  • Then run: ./setup.sh  (installs Python deps, fetches mathlib, downloads checkpoint)"

private def healthUrl (cfg : RuntimeConfig) : String :=
  s!"http://{cfg.host}:{cfg.port}/health"

private def inferUrl (cfg : RuntimeConfig) : String :=
  s!"http://{cfg.host}:{cfg.port}/infer"

private def parseAsTacticSeq (env : Environment) (input : String) (fileName := "<satp>") :
    Except String (TSyntax ``tacticSeq) :=
  match Lean.Parser.runParserCategory env `tactic input.trim fileName with
  | .ok stx => .ok ⟨stx⟩
  | .error err => .error err

/--
Render the current tactic state as a theorem statement close to SATP's
training distribution.

Local `let` declarations are zeta-reduced in Lean before pretty-printing,
which avoids Python-side reconstruction pitfalls such as turning
`let b := n / 6` into an `optParam` binder.
-/
private def collectFormalStatement : TacticM String := withMainContext do
  let mut binderLines : Array String := #[]
  for decl in ← getLCtx do
    if decl.userName.isAnonymous
        || decl.isImplementationDetail
        || decl.binderInfo.isInstImplicit
        || decl.isLet then
      continue
    let renderedType ← ppExpr (← zetaReduce decl.type)
    binderLines := binderLines.push s!"  ({decl.userName} : {renderedType.pretty})"

  let renderedGoal ← ppExpr (← zetaReduce (← getMainTarget))
  let lines := #["theorem _satpGoal"] ++ binderLines ++ #[s!"  : {renderedGoal.pretty} := by"]
  return "\n".intercalate lines.toList

private def elabUserLemmaNames (terms : Array (TSyntax `term)) : TacticM (Array String) := do
  let mut names := #[]
  for term in terms do
    match term with
    | `(term| $id:ident) =>
      names := names.push id.getId.toString
    | _ =>
      throwError "satp only supports identifier lemmas in [ ... ], got: {term}"
  pure names

private def checkServerHealth (cfg : RuntimeConfig) : IO Bool := do
  let out ← IO.Process.output {
    cmd := "curl"
    args := #[
      "-sS",
      "--max-time",
      "2",
      healthUrl cfg
    ]
  }
  if out.exitCode != 0 then
    return false
  match Json.parse out.stdout with
  | .error _ => return false
  | .ok payload =>
    match payload.getObjValAs? Bool "ok" with
    | .ok ok => return ok
    | .error _ => return false

private def commandExists (name : String) : IO Bool := do
  try
    let out ← IO.Process.output { cmd := "which", args := #[name] }
    pure (out.exitCode == 0)
  catch _ =>
    pure false

private def pickServiceRunner (cfg : RuntimeConfig) (mode : String) : IO (Option ServiceRunner) := do
  for runner in serviceRunners cfg mode do
    if ← commandExists runner.probe then
      return some runner
  return none

private def spawnService (cfg : RuntimeConfig) (runner : ServiceRunner) : IO Unit := do
  let _ ← IO.Process.spawn {
    cmd := runner.cmd
    args := runner.args
    cwd := some cfg.repoRoot
    stdin := .null
    stdout := .null
    stderr := .null
    setsid := true
  }
  pure ()

private partial def waitForServer (cfg : RuntimeConfig) (attempts : Nat := 240) : IO Bool := do
  if attempts == 0 then
    return false
  if ← checkServerHealth cfg then
    return true
  IO.sleep 250
  waitForServer cfg (attempts - 1)

private def ensureServerRunning (cfg : RuntimeConfig) : TacticM (Except MessageData Unit) := do
  let healthy : Bool ← liftM (m := IO) <| checkServerHealth cfg
  if healthy then
    return .ok ()
  let runner? ← liftM (m := IO) <| pickServiceRunner cfg "--serve"
  let some runner := runner?
    | return .error (pythonEnvironmentMessage cfg)
  try
    let _ ← liftM (m := IO) <| spawnService cfg runner
  catch _ =>
    return .error (m!"satp: failed to start inference service\n" ++
      m!"  • Command: {describeRunner runner}\n" ++
      m!"  • Try running `./setup.sh` in {cfg.repoRoot}")
  let ready : Bool ← liftM (m := IO) <| waitForServer cfg
  if ready then
    return .ok ()
  return .error (
    m!"satp: inference service did not respond (timeout after 60s)\n" ++
    m!"  • Service command: {describeRunner runner}\n" ++
    m!"  • Possible causes:\n" ++
    m!"    1. Missing checkpoint — run: ./setup.sh in {cfg.repoRoot}\n" ++
    m!"    2. Checkpoint path mismatch — expected: {cfg.checkpoint}\n" ++
    m!"    3. Missing Python deps or mathlib deps — run: ./setup.sh in {cfg.repoRoot}\n" ++
    m!"    4. Port conflict — check if port {cfg.port} is in use: lsof -i :{cfg.port}")


private def callInferenceService
    (cfg : RuntimeConfig)
    (formalStatement : String)
    (userLemmas : Array String) : TacticM (Except MessageData String) := do
  let requestBody := Json.compress <| Json.mkObj [
    ("formal_statement", Json.str formalStatement),
    ("user_lemmas", Json.arr <| userLemmas.map Json.str),
    ("tactic_name", Json.str "aesop")
  ]
  let out ← liftM (m := IO) <| IO.Process.output {
    cmd := "curl"
    args := #[
      "-sS",
      "--max-time",
      toString cfg.requestTimeout,
      -- Retry on transient transport failures (connection reset,
      -- resolve fail, curl-level timeout).  curl 7.71+ has
      -- `--retry-all-errors` which extends retry to HTTP 5xx, but this
      -- host's curl is 7.68 (no such flag) and the whole `curl` call
      -- fails immediately if we pass it.  Skip that flag: HTTP 503 /
      -- 500 retry is handled one level up by the wrapper's second
      -- `((satp?); done)` branch in prove_aesop.py.
      "--retry", "3",
      "--retry-delay", "1",
      "--retry-max-time", "180",
      "-H",
      "Content-Type: application/json",
      "-X",
      "POST",
      inferUrl cfg,
      "-d",
      requestBody
    ]
  }
  if out.exitCode != 0 then
    return .error m!"satp inference request failed: {out.stderr}"
  let payload ←
    match Json.parse out.stdout with
    | .ok payload => pure payload
    | .error err =>
        return .error m!"satp returned invalid JSON: {err}\n{out.stdout}"
  return match payload.getObjValAs? Bool "ok" with
  | .ok true =>
    match payload.getObjValAs? String "tactic" with
    | .ok tactic => .ok tactic
    | .error err => .error m!"satp response missing tactic string: {err}"
  | .ok false =>
    match payload.getObjValAs? String "error" with
    | .ok err => .error m!"satp inference failed: {err}"
    | .error _ => .error m!"satp inference failed without an error message"
  | .error err =>
    .error m!"satp response missing ok field: {err}"

private def parseReturnedTactic
    (tacticString : String) : TacticM (Except MessageData (TSyntax ``tacticSeq)) := do
  match parseAsTacticSeq (← getEnv) tacticString with
  | .error err =>
    return .error m!"satp produced an unparsable tactic:\n{tacticString}\n\n{err}"
  | .ok tacticSeq =>
    return .ok tacticSeq

private def evalReturnedTactic
    (stxRef : Syntax)
    (tacticSeq : TSyntax ``tacticSeq)
    (traceScript : Bool := false) : TacticM (Except MessageData Unit) := do
  try
    evalTactic tacticSeq
    if traceScript then
      Aesop.addTryThisTacticSeqSuggestion stxRef tacticSeq (← getRef)
    return .ok ()
  catch err =>
    return .error err.toMessageData

private def fallbackToAesop
    (stxRef : Syntax)
    (reason : MessageData)
    (traceScript : Bool := false) : TacticM Unit := do
  logWarning m!"satp fallback to plain aesop: {reason}"
  let tac ← if traceScript then `(tactic| aesop?) else `(tactic| aesop)
  withRef stxRef (evalTactic tac)

private def runSatpWithFallback
    (stxRef : Syntax)
    (cfg : RuntimeConfig)
    (lemmaNames : Array String)
    (traceScript : Bool := false) : TacticM Unit := do
  match ← ensureServerRunning cfg with
  | .error reason =>
      fallbackToAesop stxRef reason traceScript
      return
  | .ok () =>
      pure ()
  let formalStatement ←
    try
      collectFormalStatement
    catch err =>
      fallbackToAesop stxRef m!"satp failed to render theorem state: {err.toMessageData}" traceScript
      return
  match ← callInferenceService cfg formalStatement lemmaNames with
  | .error reason =>
      fallbackToAesop stxRef reason traceScript
  | .ok tacticString =>
      match ← parseReturnedTactic tacticString with
      | .error reason =>
          fallbackToAesop stxRef reason traceScript
      | .ok tacticSeq =>
          match ← evalReturnedTactic stxRef tacticSeq traceScript with
          | .ok () => pure ()
          | .error reason => fallbackToAesop stxRef reason traceScript

/--
`satp` (Steering Aesop for Theorem Proving) queries a local SATP inference
service for a tailored `aesop` configuration and runs it inside Lean.

Usage:
- `satp` — run the model-backed pipeline on the current goal.
- `satp [lem₁, lem₂, …]` — append the listed identifiers as extra `aesop`
  unsafe rules before invocation.

If the Python service is unavailable or returns a tactic that fails to parse
or discharge the goal, `satp` logs one warning and falls back to plain `aesop`.
The server endpoint is `127.0.0.1:5177` by default; override with
`SATP_SERVER_HOST` / `SATP_SERVER_PORT` / `SATP_REQUEST_TIMEOUT` env vars.
-/
syntax (name := satp) "satp" (ppSpace "[" (term),* "]")? : tactic

/--
`satp?` behaves like `satp` but also prints the exact tactic it ran as a
"Try this" suggestion so you can replace the call with a concrete proof.
If `satp?` falls back to `aesop`, the fallback is `aesop?` and the
suggestion comes from there.
-/
syntax (name := satpTacticQuery) "satp?" (ppSpace "[" (term),* "]")? : tactic

private def runSatpFromSyntax
    (stxRef : Syntax) (terms : Array (TSyntax `term)) (traceScript : Bool) :
    TacticM Unit := do
  let cfg ← liftM (m := IO) runtimeConfigFromEnv
  let lemmaNames ← elabUserLemmaNames terms
  runSatpWithFallback stxRef cfg lemmaNames traceScript

@[tactic satp]
def evalSatp : Tactic
  | `(tactic| satp%$stxRef [$terms,*]) => runSatpFromSyntax stxRef terms false
  | `(tactic| satp%$stxRef)             => runSatpFromSyntax stxRef #[] false
  | _ => throwUnsupportedSyntax

@[tactic satpTacticQuery]
def evalSatpQuery : Tactic
  | `(tactic| satp?%$stxRef [$terms,*]) => runSatpFromSyntax stxRef terms true
  | `(tactic| satp?%$stxRef)             => runSatpFromSyntax stxRef #[] true
  | _ => throwUnsupportedSyntax

end LeanSATP
