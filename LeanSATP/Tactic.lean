import Aesop
import Lean

open Lean Parser Elab Tactic Meta
open System (FilePath)

namespace LeanSATP

private def defaultCheckpoint : String :=
  "hf://ChristianZ97/SATP-aesop-policy/best_checkpoint.pt"

private def defaultServerHost : String := "127.0.0.1"
private def defaultServerPort : Nat := 5177
private def defaultRequestTimeout : Nat := 30

private partial def ascend (path : FilePath) (steps : Nat) : FilePath :=
  if steps == 0 then
    path
  else
    match path.parent with
    | some parent => ascend parent (steps - 1)
    | none => path

private def packageRoot : IO FilePath := do
  let oleanPath ← Lean.findOLean `LeanSATP.Tactic
  return (ascend oleanPath 6).normalize

private def defaultRepoRoot : IO FilePath := do
  Lean.realPathNormalized (← packageRoot)

private def defaultCacheDir : IO String := do
  return (((← defaultRepoRoot) / "cache").normalize.toString)

structure RuntimeConfig where
  repoRoot : FilePath
  cacheDir : String
  checkpoint : String

structure ServiceRunner where
  probe : String
  cmd : String
  args : Array String

private def envOrDefault (key fallback : String) : IO String := do
  return (← IO.getEnv key).getD fallback

private def runtimeConfigFromEnv : IO RuntimeConfig := do
  let repoRoot ← defaultRepoRoot
  return {
    repoRoot := repoRoot
    cacheDir := ← envOrDefault "SATP_CACHE_DIR" (← defaultCacheDir)
    checkpoint := defaultCheckpoint
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
  m!"  • Then run: uv sync  (installs dependencies and downloads checkpoint)"

private def healthUrl : String :=
  s!"http://{defaultServerHost}:{defaultServerPort}/health"

private def inferUrl : String :=
  s!"http://{defaultServerHost}:{defaultServerPort}/infer"

private def parseAsTacticSeq (env : Environment) (input : String) (fileName := "<satp>") :
    Except String (TSyntax ``tacticSeq) :=
  let parser := andthenFn whitespace Lean.Parser.Tactic.tacticSeq.fn
  let ictx := mkInputContext input fileName
  let state := parser.run ictx { env, options := {} } (getTokenTable env) (mkParserState input)
  if state.hasError then
    .error (state.toErrorMsg ictx)
  else if state.pos.atEnd input then
    .ok ⟨state.stxStack.back⟩
  else
    .error ((state.mkError "end of input").toErrorMsg ictx)

private def hypothesisJson (decl : LocalDecl) : MetaM Json := do
  let declType ← instantiateMVars decl.type
  let renderedType ← ppExpr declType
  pure <| Json.mkObj [
    ("name", Json.str decl.userName.toString),
    ("type", Json.str renderedType.pretty),
  ]

private def collectHypotheses : TacticM (Array Json) := withMainContext do
  let lctx ← getLCtx
  let mut hyps := #[]
  for decl in lctx do
    if decl.isImplementationDetail then
      continue
    if decl.userName.isAnonymous then
      continue
    hyps := hyps.push (← hypothesisJson decl)
  pure hyps

private def collectGoal : TacticM String := withMainContext do
  let goal ← instantiateMVars (← getMainTarget)
  return (← ppExpr goal).pretty

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

private partial def waitForServer (attempts : Nat := 80) : IO Bool := do
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
    return .error m!"satp: failed to start inference service\n" ++
      m!"  • Command: {describeRunner runner}\n" ++
      m!"  • Try running `uv sync` in {cfg.repoRoot} to reinstall dependencies"
  let ready : Bool ← liftM (m := IO) waitForServer
  if ready then
    return .ok ()
  return .error
    m!"satp: inference service did not respond (timeout after 20s)\n" ++
    m!"  • Service command: {describeRunner runner}\n" ++
    m!"  • Possible causes:\n" ++
    m!"    1. Missing checkpoint — run: uv sync\n" ++
    m!"    2. Checkpoint path mismatch — expected: {cfg.checkpoint}\n" ++
    m!"    3. Missing Python deps — run: uv sync\n" ++
    m!"    4. Port conflict — check if port 5177 is already in use: lsof -i :5177"

private def callInferenceService
    (_cfg : RuntimeConfig)
    (goal : String)
    (hypotheses : Array Json)
    (userLemmas : Array String) : TacticM (Except MessageData String) := do
  let requestBody := Json.compress <| Json.mkObj [
    ("goal", Json.str goal),
    ("hypotheses", Json.arr hypotheses),
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

private def evalReturnedTactic (tacticString : String) : TacticM (Except MessageData Unit) := do
  match parseAsTacticSeq (← getEnv) tacticString with
  | .error err =>
    return .error m!"satp produced an unparsable tactic:\n{tacticString}\n\n{err}"
  | .ok tacticSeq =>
    try
      evalTactic tacticSeq
      return .ok ()
    catch err =>
      return .error err.toMessageData

private def fallbackToDefaultAesop (reason : MessageData) : TacticM Unit := do
  logWarning m!"satp fallback to plain aesop: {reason}"
  evalTactic (← `(tactic| aesop))

private def runSatpWithFallback (cfg : RuntimeConfig) (lemmaNames : Array String) : TacticM Unit := do
  match ← ensureServerRunning cfg with
  | .error reason =>
      fallbackToDefaultAesop reason
      return
  | .ok () =>
      pure ()
  let goal ← collectGoal
  let hyps ← collectHypotheses
  match ← callInferenceService cfg goal hyps lemmaNames with
  | .error reason =>
      fallbackToDefaultAesop reason
  | .ok tacticString =>
      match ← evalReturnedTactic tacticString with
      | .ok () => pure ()
      | .error reason => fallbackToDefaultAesop reason

private def bestEffortDownloadOnImport : IO Unit := do
  if (← IO.getEnv "SATP_SKIP_IMPORT_DOWNLOAD").isSome then
    return
  let cfg ← runtimeConfigFromEnv
  unless (← cfg.repoRoot.pathExists) do
    IO.eprintln s!"[LeanSATP] Skipping checkpoint prefetch; repo root not found: {cfg.repoRoot}"
    return
  let runner? ← pickServiceRunner cfg "--download-only"
  let some runner := runner?
    | return
  try
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
  catch _ =>
    pure ()

initialize
  discard <| bestEffortDownloadOnImport

syntax (name := satp) "satp" (ppSpace "[" (term),* "]")? : tactic

@[tactic satp]
def evalSatp : Tactic
  | `(tactic| satp [$terms,*]) => do
      let cfg ← liftM (m := IO) runtimeConfigFromEnv
      let lemmaNames ← elabUserLemmaNames terms
      runSatpWithFallback cfg lemmaNames
  | `(tactic| satp) => do
      let cfg ← liftM (m := IO) runtimeConfigFromEnv
      runSatpWithFallback cfg #[]
  | _ => throwUnsupportedSyntax

end LeanSATP
