import Aesop
import Lean

open Lean Parser Elab Tactic Meta
open System (FilePath)

namespace LeanSATP

private def defaultServerHost : String := "127.0.0.1"
private def defaultServerPort : Nat := 5177
private def defaultRequestTimeout : Nat := 30

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
  return {
    repoRoot := repoRoot
    cacheDir := cacheDir
    checkpoint := checkpointEnv?.getD (defaultCheckpointFromCacheDir cacheDir)
  }

private def serviceModuleArgs (cfg : RuntimeConfig) (mode : String) : Array String :=
  let args := #[
    "-m",
    "leansatp_runtime.service",
    mode,
    "--checkpoint",
    cfg.checkpoint,
    "--cache-dir",
    cfg.cacheDir,
  ]
  if mode = "--serve" then
    args ++ #[
      "--host",
      defaultServerHost,
      "--port",
      toString defaultServerPort,
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
    },
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

private def healthUrl : String :=
  s!"http://{defaultServerHost}:{defaultServerPort}/health"

private def inferUrl : String :=
  s!"http://{defaultServerHost}:{defaultServerPort}/infer"

private def parseAsTacticSeq (env : Environment) (input : String) (fileName := "<satp>") :
    Except String (TSyntax ``tacticSeq) :=
  match Lean.Parser.runParserCategory env `tactic input.trim fileName with
  | .ok stx => .ok ⟨stx⟩
  | .error err => .error err

private def applyLetSubst (letFVars letValues : Array Expr) (e : Expr) : Expr :=
  if letFVars.isEmpty then
    e
  else
    e.replaceFVars letFVars letValues

/--
Render the current tactic state as a theorem statement close to SATP's
training distribution.

Local `let` declarations are substituted away in Lean before pretty-printing,
which avoids Python-side reconstruction pitfalls such as turning
`let b := n / 6` into an `optParam` binder.
-/
private def collectFormalStatement : TacticM String := withMainContext do
  let lctx ← getLCtx
  let mut letFVars : Array Expr := #[]
  let mut letValues : Array Expr := #[]
  let mut binderLines : Array String := #[]

  for decl in lctx do
    if decl.userName.isAnonymous then
      continue

    let declType ← instantiateMVars decl.type
    let declType := applyLetSubst letFVars letValues declType

    if let some value := decl.value? then
      let value ← instantiateMVars value
      let value := applyLetSubst letFVars letValues value
      letFVars := letFVars.push decl.toExpr
      letValues := letValues.push value
      continue

    if decl.isImplementationDetail || decl.binderInfo.isInstImplicit then
      continue

    let renderedType ← ppExpr declType
    binderLines := binderLines.push s!"  ({decl.userName} : {renderedType.pretty})"

  let goal ← instantiateMVars (← getMainTarget)
  let goal := applyLetSubst letFVars letValues goal
  let renderedGoal ← ppExpr goal

  let mut lines : Array String := #[s!"theorem satp_goal"]
  lines := lines ++ binderLines
  lines := lines.push s!"  : {renderedGoal.pretty} := by"
  return String.intercalate "\n" lines.toList

private def elabUserLemmaNames (terms : Array (TSyntax `term)) : TacticM (Array String) := do
  let mut names := #[]
  for term in terms do
    match term with
    | `(term| $id:ident) =>
      names := names.push id.getId.toString
    | _ =>
      throwError "satp only supports identifier lemmas in [ ... ], got: {term}"
  pure names

private def checkServerHealth : IO Bool := do
  let out ← IO.Process.output {
    cmd := "curl"
    args := #[
      "-sS",
      "--max-time",
      "2",
      healthUrl,
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

private partial def waitForServer (attempts : Nat := 240) : IO Bool := do
  if attempts == 0 then
    return false
  if ← checkServerHealth then
    return true
  IO.sleep 250
  waitForServer (attempts - 1)

private def ensureServerRunning (cfg : RuntimeConfig) : TacticM (Except MessageData Unit) := do
  let healthy : Bool ← liftM (m := IO) checkServerHealth
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
  let ready : Bool ← liftM (m := IO) waitForServer
  if ready then
    return .ok ()
  return .error (
    m!"satp: inference service did not respond (timeout after 20s)\n" ++
    m!"  • Service command: {describeRunner runner}\n" ++
    m!"  • Possible causes:\n" ++
    m!"    1. Missing checkpoint — run: ./setup.sh\n" ++
    m!"    2. Checkpoint path mismatch — expected: {cfg.checkpoint}\n" ++
    m!"    3. Missing Python deps or mathlib deps — run: ./setup.sh\n" ++
    m!"    4. Port conflict — check if port 5177 is in use: lsof -i :5177")


private def callInferenceService
    (_cfg : RuntimeConfig)
    (formalStatement : String)
    (userLemmas : Array String) : TacticM (Except MessageData String) := do
  let requestBody := Json.compress <| Json.mkObj [
    ("formal_statement", Json.str formalStatement),
    ("user_lemmas", Json.arr <| userLemmas.map Json.str),
    ("tactic_name", Json.str "aesop"),
  ]
  let out ← liftM (m := IO) <| IO.Process.output {
    cmd := "curl"
    args := #[
      "-sS",
      "--max-time",
      toString defaultRequestTimeout,
      "-H",
      "Content-Type: application/json",
      "-X",
      "POST",
      inferUrl,
      "-d",
      requestBody,
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
  withRef stxRef do
    if traceScript then
      evalTactic (← `(tactic| aesop?))
    else
      evalTactic (← `(tactic| aesop))

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

syntax (name := satp) "satp" (ppSpace "[" (term),* "]")? : tactic
syntax (name := satpTacticQuery) "satp?" (ppSpace "[" (term),* "]")? : tactic

@[tactic satp]
def evalSatp : Tactic
  | `(tactic| satp%$stxRef [$terms,*]) => do
      let cfg ← liftM (m := IO) runtimeConfigFromEnv
      let lemmaNames ← elabUserLemmaNames terms
      runSatpWithFallback stxRef cfg lemmaNames
  | `(tactic| satp%$stxRef) => do
      let cfg ← liftM (m := IO) runtimeConfigFromEnv
      runSatpWithFallback stxRef cfg #[]
  | _ => throwUnsupportedSyntax

@[tactic satpTacticQuery]
def evalSatpQuery : Tactic
  | `(tactic| satp?%$stxRef [$terms,*]) => do
      let cfg ← liftM (m := IO) runtimeConfigFromEnv
      let lemmaNames ← elabUserLemmaNames terms
      runSatpWithFallback stxRef cfg lemmaNames (traceScript := true)
  | `(tactic| satp?%$stxRef) => do
      let cfg ← liftM (m := IO) runtimeConfigFromEnv
      runSatpWithFallback stxRef cfg #[] (traceScript := true)
  | _ => throwUnsupportedSyntax

end LeanSATP
