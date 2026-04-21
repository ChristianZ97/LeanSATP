import Aesop
import Lean
import Mathlib.Tactic.ClearExcept

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
Iterate the filtered local context used by both input renderers. Keeps
non-anonymous, non-implementation, non-inst-implicit, non-`let` hypotheses
and zeta-reduces their types so Python reconstruction doesn't trip on
let-bindings (e.g. `let b := n / 6` turning into an `optParam` binder).
-/
private def collectFilteredHyps : TacticM (Array (Lean.Name × Lean.Format)) :=
  withMainContext do
    let mut acc : Array (Lean.Name × Lean.Format) := #[]
    for decl in ← getLCtx do
      if decl.userName.isAnonymous
          || decl.isImplementationDetail
          || decl.binderInfo.isInstImplicit
          || decl.isLet then
        continue
      let renderedType ← ppExpr (← zetaReduce decl.type)
      acc := acc.push (decl.userName, renderedType)
    return acc

/--
Render the current tactic state as a theorem statement close to SATP's
original (2025) training distribution.
-/
private def collectFormalStatement : TacticM String := withMainContext do
  let hyps ← collectFilteredHyps
  let binderLines := hyps.map fun (n, t) => s!"  ({n} : {t.pretty})"
  let renderedGoal ← ppExpr (← zetaReduce (← getMainTarget))
  let lines := #["theorem _satpGoal"] ++ binderLines ++ #[s!"  : {renderedGoal.pretty} := by"]
  return "\n".intercalate lines.toList

/--
Render the current tactic state as a raw goal state (Lean's `⊢` format),
matching the pretrain distribution and the input modality used by
BFS-Prover / ReProver / LeanCopilot. One hypothesis per line, goal
prefixed with `⊢`.
-/
private def collectGoalState : TacticM String := withMainContext do
  let hyps ← collectFilteredHyps
  let hypLines := hyps.map fun (n, t) => s!"{n} : {t.pretty}"
  let renderedGoal ← ppExpr (← zetaReduce (← getMainTarget))
  let lines := hypLines ++ #[s!"⊢ {renderedGoal.pretty}"]
  return "\n".intercalate lines.toList

/--
Pick the input renderer based on `SATP_INPUT_MODE` env var (read once per
call; cost is negligible vs the HTTP round-trip).
  • `goal` (default) → `collectGoalState` — aligns with pretrain + peer
    tacGens (BFS-Prover/ReProver/LeanCopilot).
  • `theorem` → `collectFormalStatement` — original 2025 SATP finetune
    format. Keep for A/B revert without rebuild.
-/
private def collectSatpInput : TacticM String := do
  let mode := (← IO.getEnv "SATP_INPUT_MODE").getD "goal"
  if mode == "theorem" then
    collectFormalStatement
  else
    collectGoalState

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
    (userLemmas : Array String)
    (stripRetrieval : Bool := false)
    (hintPriority : Nat := 40) : TacticM (Except MessageData String) := do
  -- Extra fields on top of the legacy (formal_statement / user_lemmas /
  -- tactic_name) contract drive the satp? 3-layer cascade:
  --   • strip_retrieval = true  → server drops the policy's retrieval-
  --     sourced lemma rules (Layer 1 "trust sketch" branch)
  --   • hint_priority          → priority_pct used when appending
  --     user_lemmas as aesop rules (50 for cascade layers that carry
  --     hints, 40 for the legacy code path)
  let requestBody := Json.compress <| Json.mkObj [
    ("formal_statement", Json.str formalStatement),
    ("user_lemmas", Json.arr <| userLemmas.map Json.str),
    ("tactic_name", Json.str "aesop"),
    ("strip_retrieval", Json.bool stripRetrieval),
    ("hint_priority", toJson hintPriority)
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
    (tacticSeq : TSyntax ``tacticSeq) : TacticM (Except MessageData Unit) := do
  try
    evalTactic tacticSeq
    return .ok ()
  catch err =>
    return .error err.toMessageData

-- Emit a single, positionally-anchored Try-this suggestion for a cascade
-- layer that just closed the goal. The message is a `logInfoAt stxRef`
-- diagnostic so the lake log carries a `file:line:col: info:` prefix
-- uniquely identifying the satp? call site — callers pair bodies to
-- source gaps by line rather than by emission order (which breaks when
-- a layer eval-ok but done-fails, or a later layer succeeds).
--
-- `layerLabel` is one of "bare" / "L1" / "L2" / "L3" / "fallback" so
-- downstream analysis can distinguish which cascade branch fired.
-- `emitClear = true` prepends the corresponding `clear * -` (possibly
-- with keep-list) to the aesop body — reflecting what the layer
-- actually ran inside tryCascadeLayer before calling the policy.
private def emitLayerSuggestion
    (stxRef : Syntax)
    (layerLabel : String)
    (clearKeep : Array String)
    (emitClear : Bool)
    (tacticText : String) : TacticM Unit := do
  let keepStr : String := String.intercalate " " clearKeep.toList
  let clearPrefix : String :=
    if !emitClear then ""
    else if clearKeep.isEmpty then "clear * -\n  "
    else s!"clear * - {keepStr}\n  "
  logInfoAt stxRef m!"satp\nlayer={layerLabel}\nTry this:\n  {clearPrefix}{tacticText}"

private def fallbackToAesop
    (stxRef : Syntax)
    (reason : MessageData)
    (traceScript : Bool := false) : TacticM Unit := do
  logWarning m!"satp fallback to plain aesop: {reason}"
  withRef stxRef (evalTactic (← `(tactic| aesop)))
  if traceScript then
    -- Fallback closed the goal live via plain `aesop`, but reproducing
    -- that outcome standalone (no retrieval state) is unreliable.
    -- Emit `sorry` as the honest suggestion so downstream Form B/C
    -- under-claims rather than pretending bare `aesop` will pass.
    emitLayerSuggestion stxRef "fallback" #[] false "sorry"

-- Clear all local hypotheses except the listed identifier names (Mathlib
-- `clear * -` syntax).  When `keep` is empty this clears everything that
-- can be safely cleared (signature variables referenced by the goal stay).
private def clearExceptHints (keep : Array String) : TacticM Unit := do
  let idents := keep.map (fun name => mkIdent (Name.mkSimple name))
  if idents.isEmpty then
    evalTactic (← `(tactic| clear * -))
  else
    evalTactic (← `(tactic| clear * - $idents*))

-- Run one HTTP /infer request with the given flags, then parse & eval the
-- returned tactic and check that no goals remain.  Returns `true` on total
-- success; `false` if any step fails (caller is expected to have wrapped
-- this in `saveState` + `restoreState` so tactic state is reset).
private def tryInferLayer
    (stxRef : Syntax)
    (cfg : RuntimeConfig)
    (userLemmas : Array String)
    (stripRetrieval : Bool)
    (hintPriority : Nat)
    (traceScript : Bool)
    (layerLabel : String)
    (clearKeep : Array String)
    (emitClear : Bool) : TacticM Bool := do
  let formalStatement ←
    try collectSatpInput
    catch _ => return false
  match ← callInferenceService cfg formalStatement userLemmas stripRetrieval hintPriority with
  | .error _ => return false
  | .ok tacticString =>
      match ← parseReturnedTactic tacticString with
      | .error _ => return false
      | .ok tacticSeq =>
          match ← evalReturnedTactic tacticSeq with
          | .error _ => return false
          | .ok () =>
              -- Require all goals closed; this is what distinguishes a
              -- cascade "this layer worked" from "aesop returned without
              -- fully discharging". Only after `done` succeeds do we
              -- emit the Try-this suggestion — this avoids the
              -- eval-ok/done-fail case polluting the log with bodies
              -- that don't actually close the gap.
              try
                evalTactic (← `(tactic| done))
                if traceScript then
                  emitLayerSuggestion stxRef layerLabel clearKeep emitClear tacticString
                return true
              catch _ => return false

-- Try a single cascade layer with a specific clear + HTTP configuration.
-- Wraps the attempt in saveState / restoreState so failure rolls the
-- tactic state back (otherwise a half-applied `clear` would leak).
private def tryCascadeLayer
    (stxRef : Syntax)
    (cfg : RuntimeConfig)
    (clearKeep : Array String)
    (userLemmas : Array String)
    (stripRetrieval : Bool)
    (hintPriority : Nat)
    (traceScript : Bool)
    (layerLabel : String) : TacticM Bool := do
  let snapshot ← saveState
  try
    clearExceptHints clearKeep
    if ← tryInferLayer stxRef cfg userLemmas stripRetrieval hintPriority
        traceScript layerLabel clearKeep true then
      return true
    else
      snapshot.restore
      return false
  catch _ =>
    snapshot.restore
    return false

-- 3-layer cascade (when hints are provided) that mirrors DSP+'s
-- bfsaesopLoop escalation pattern but with SATP-specific signals:
--   L1: narrow scope to hints + strip retrieval + hint priority 50%
--       ("strongly trust the sketch; drop our own retrieval suggestions")
--   L2: clear all local context + keep retrieval + no hints passed
--       ("distrust the sketch; let the policy's retrieval decide")
--   L3: narrow scope to hints + keep retrieval + hint priority 50%
--       ("kitchen-sink: combine sketch hints with retrieval")
-- Without hints only L2 runs.  All layers fall back to `aesop`/`aesop?`
-- if none closes the goal.
private def runSatpCascade
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

  if lemmaNames.isEmpty then
    -- Bare `satp?` (no hints): single legacy-style call on the current
    -- goal state.  We deliberately do NOT clear here because `clear * -`
    -- with no keep list will drop signature-level hypotheses whenever
    -- the goal itself doesn't reference them (e.g. `h_pos : 0 < n` in a
    -- goal `1 ≤ n`), which would make otherwise-provable top-level
    -- callers fail for a reason unrelated to the policy.
    match ← tryInferLayer stxRef cfg #[] false 40 traceScript "bare" #[] false with
    | true => return
    | false =>
        fallbackToAesop stxRef m!"satp?: retrieval-only attempt failed" traceScript
        return

  -- With hints: full 3-layer cascade.  Each layer narrows scope to the
  -- named hints (keeping signature vars referenced by the goal anyway)
  -- or, for L2, strips intermediate haves.
  --
  -- L1: narrow scope + strip retrieval + hint@50%  (strongly trust sketch)
  if ← tryCascadeLayer stxRef cfg lemmaNames lemmaNames true 50 traceScript "L1" then
    return
  -- L2: clear intermediates + keep retrieval + no hints  (distrust sketch)
  if ← tryCascadeLayer stxRef cfg #[] #[] false 40 traceScript "L2" then
    return
  -- L3: narrow scope + keep retrieval + hint@50%  (combined)
  if ← tryCascadeLayer stxRef cfg lemmaNames lemmaNames false 50 traceScript "L3" then
    return

  -- All cascade paths failed — fall back to a single plain `aesop` so the
  -- caller gets a last-ditch attempt rather than `satp?` itself throwing.
  fallbackToAesop stxRef m!"satp? cascade: all layers failed" traceScript

/--
`satp` (Steering Aesop for Theorem Proving) queries a local SATP inference
service for a tailored `aesop` configuration and runs it inside Lean.

Usage:
- `satp` — retrieval-only run on the current goal (Layer 2 only).  Clears
  intermediate local hypotheses so the policy sees a theorem-level goal
  matching its training distribution.
- `satp [h₁, h₂, …]` — 3-layer cascade that respects sketch-provided
  hint hypotheses:
    L1  narrow scope to `{h₁, h₂}`, strip retrieval rules, hint at 50%
        ("strongly trust sketch; drop retrieval suggestions")
    L2  clear intermediates, keep retrieval, discard hints
        ("distrust sketch; rely on retrieval")
    L3  narrow scope to `{h₁, h₂}`, keep retrieval, hint at 50%
        ("combined; both signals on the table")
  Each layer saves/restores tactic state, so a failed layer does not leak
  `clear` effects into the next.

If the Python service is unavailable or every layer fails, `satp` logs
one warning and falls back to plain `aesop`.  The server endpoint is
`127.0.0.1:5177` by default; override with `SATP_SERVER_HOST` /
`SATP_SERVER_PORT` / `SATP_REQUEST_TIMEOUT` env vars.
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
  runSatpCascade stxRef cfg lemmaNames traceScript

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
