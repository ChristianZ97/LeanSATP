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
  • `theorem` (default, 2026-04-21) → `collectFormalStatement` — original
    2025 SATP finetune format. A/B on minif2f-test showed
    theorem-level 103/244 vs goal-state 97/244 (overlap 95,
    goal-only 2, theorem-only 8), so theorem stays the default.
  • `goal` → `collectGoalState` — aligns with pretrain + peer tacGens
    (BFS-Prover/ReProver/LeanCopilot).  Kept as opt-in for future
    gap-level finetune experiments where the model is retrained on
    bare goal-state inputs.

Retrying the alternate mode before failing is intentionally
NOT implemented: the retry would double HTTP latency for every failed
policy call, and per-call `satp_input_mode=...` telemetry already gives
downstream analysis the signal to compute per-mode pass rate post-hoc.
If a future ablation needs it, wire a second `tryInferLayer` call with
the alternate mode between `snap.restore` and the trailing `throwError`
in `runSatpCascade`.
-/
private def collectSatpInput : TacticM String := do
  let mode := (← IO.getEnv "SATP_INPUT_MODE").getD "theorem"
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
  -- Surface the discard so callers
  -- don't silently rely on hint threading that no longer reaches the
  -- model or aesop rules. Compile-time warning is visible in IDE/CI
  -- without hard-erroring on existing call sites (pipeline wrappers
  -- were already switched to bare `satp?` in build_stages.py +
  -- tactic_providers.py, so this only fires on hand-written callers).
  if !names.isEmpty then
    Lean.logWarning m!"satp: bracketed hint list {names} is accepted for API back-compat but currently discarded (policy model does not ingest hints, and no aesop-rule injection is performed). See LeanSATP/README.md for migration guidance."
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
-- `layerLabel` is "policy" for the single-shot policy call; retained as
-- a param for forward compatibility with any re-introduced cascading.
-- `emitClear = true` prepends the corresponding `clear * -` (possibly
-- with keep-list) to the aesop body — reflecting what the layer
-- actually ran before calling the policy (retained for API symmetry;
-- the single-shot cascade always passes `emitClear=false`).
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

-- User-facing fallback for bare `satp`: on server / policy failure, run
-- plain `aesop` and log a warning. NOT used by `satp?` — the ?-variant
-- is the eval-side tactic and must throw on policy-fail so paper's
-- `satp` mode closure counts reflect the 0.2B policy alone, not aesop
-- bleed through. See `runSatpCascade`'s traceScript branching.
private def fallbackToAesop
    (stxRef : Syntax)
    (reason : MessageData) : TacticM Unit := do
  logWarning m!"satp fallback to plain aesop: {reason}"
  withRef stxRef (evalTactic (← `(tactic| aesop)))

-- Structured failure kinds for policy attempts. Splits "SATP miss" from
-- infrastructure/protocol faults so post-hoc analysis can filter cleanly:
-- `miss` is a genuine policy output that didn't close the goal (counts
-- as a true SATP negative for the paper matrix); the others indicate
-- server / network / server-side / parse / tactic-exec problems that
-- should be retried or investigated rather than silently counted as
-- SATP misses.
inductive SatpFailKind where
  | input    (msg : MessageData)   -- could not collect goal/theorem input
  | request  (msg : MessageData)   -- /infer HTTP call itself failed
  | parse    (msg : MessageData)   -- returned text couldn't be parsed
  | exec     (msg : MessageData)   -- tactic parsed but elaboration threw
  | miss                           -- tactic ran cleanly but did not close

private def SatpFailKind.label : SatpFailKind → String
  | .input _   => "input"
  | .request _ => "request"
  | .parse _   => "parse"
  | .exec _    => "exec"
  | .miss      => "miss"

private def SatpFailKind.reason : SatpFailKind → MessageData
  | .input msg   => msg
  | .request msg => msg
  | .parse msg   => msg
  | .exec msg    => msg
  | .miss        => m!"policy tactic did not close the goal"

-- Run one HTTP /infer request with the given flags, then parse & eval the
-- returned tactic and check that no goals remain. Returns `.ok ()` on
-- total success; `.error <kind>` otherwise, where the kind distinguishes
-- a real SATP miss from server / transport / parse / exec faults.
-- Caller is expected to have wrapped this in `saveState` + `restoreState`
-- so tactic state is reset on any non-success outcome.
private def tryInferLayer
    (stxRef : Syntax)
    (cfg : RuntimeConfig)
    (userLemmas : Array String)
    (stripRetrieval : Bool)
    (hintPriority : Nat)
    (traceScript : Bool)
    (layerLabel : String)
    (clearKeep : Array String)
    (emitClear : Bool) : TacticM (Except SatpFailKind Unit) := do
  let formalStatement ←
    try collectSatpInput
    catch e => return .error (.input e.toMessageData)
  match ← callInferenceService cfg formalStatement userLemmas stripRetrieval hintPriority with
  | .error reason => return .error (.request reason)
  | .ok tacticString =>
      match ← parseReturnedTactic tacticString with
      | .error reason => return .error (.parse reason)
      | .ok tacticSeq =>
          match ← evalReturnedTactic tacticSeq with
          | .error reason => return .error (.exec reason)
          | .ok () =>
              -- Require all goals closed; this is what distinguishes a
              -- "tactic ran successfully" from "policy genuinely missed".
              -- Only after `done` succeeds do we emit the Try-this
              -- suggestion — avoids polluting the log with non-closing
              -- bodies.
              try
                evalTactic (← `(tactic| done))
                if traceScript then
                  emitLayerSuggestion stxRef layerLabel clearKeep emitClear tacticString
                return .ok ()
              catch _ => return .error .miss

-- Single-shot policy call. Model sees only the goal; `lemmaNames` is
-- discarded here (see note below).
--
-- Failure handling is gated by `traceScript`, which also distinguishes
-- the two public tactics:
--   * `satp?` (traceScript=true, eval-side)  — throw on any failure so
--     the outer `first | ((...; satp?); done) | ...` wrapper takes the
--     next alternative. No aesop fallback: that would pull Mathlib's
--     `@[aesop]` attrs from `default`, inflating `satp` mode's closure
--     count past `aesop_plain`'s and muddying the paper's 0.2B policy
--     claim.
--   * `satp`  (traceScript=false, user-facing) — fall back to plain
--     `aesop` and keep elaborating. README documents this as the
--     ergonomic entry point; LeanSATPTest uses bare `satp` without
--     any outer combinator. A hard throw here would break README
--     examples on cold server start / transient service hiccups.
--
-- The saveState/restoreState wrapper matters because `tryInferLayer`
-- may eval a multi-step aesop config that applies partial mutations
-- (e.g. `intro x; cases h; ...`) before its trailing `done` fails.
-- Without the restore, outer alternatives (or the fallback `aesop`)
-- would see a half-mutated state rather than the original goal.
--
-- Timing emit (`satp_layer=policy,ms=T,status=ok|fail`) lives OUTSIDE
-- the restored region so the fail record survives `snap.restore`
-- (Tactic.SavedState.restore also restores the message log).
-- `logInfoAt stxRef` anchors the record to the source call site so
-- downstream build_stages.py can tie timings to a specific `satp?` gap.
--
-- `lemmaNames` arrives from `satp? [h1, h2]` syntax but is intentionally
-- discarded before reaching the server: the policy model never ingests
-- user_lemmas (they were only ever appended post-inference as aesop
-- rules), and we no longer inject aesop rules manually — we want the
-- model's original behavior. Syntax is preserved for API back-compat;
-- any ident list is parsed then dropped here. If a hint-injection
-- pathway is reintroduced (e.g. feeding names to model input via a
-- separate server field), switch the `#[]` below to `lemmaNames`.
private def runSatpCascade
    (stxRef : Syntax)
    (cfg : RuntimeConfig)
    (lemmaNames : Array String)
    (traceScript : Bool := false) : TacticM Unit := do
  let _ := lemmaNames
  -- Emit the resolved input-rendering mode once per call so downstream
  -- tooling / ablation analysis can tell `theorem` and `goal` runs apart
  -- without re-reading the env. Done here, before any saveState region,
  -- so the record survives `snap.restore` on policy failure.
  let inputMode := (← IO.getEnv "SATP_INPUT_MODE").getD "theorem"
  logInfoAt stxRef s!"satp_input_mode={inputMode}"
  -- Separate event from the policy call so budget-sweep replay can
  -- account for cold-spawn overhead (health check + waitForServer can
  -- cost tens of seconds on the first call after boot; warm hits are
  -- <10ms). `ensureServerRunning` is pure IO, no tactic-state mutation,
  -- so no snapshot needed around it.
  let tEnsure0 ← IO.monoMsNow
  let ensureResult ← ensureServerRunning cfg
  let tEnsure1 ← IO.monoMsNow
  match ensureResult with
  | .error reason =>
      logInfoAt stxRef s!"satp_layer=ensure_server,ms={tEnsure1 - tEnsure0},status=fail,fail_kind=server"
      if traceScript then
        throwError "satp?: SATP server unavailable: {reason}"
      else
        fallbackToAesop stxRef reason
        return
  | .ok () =>
      logInfoAt stxRef s!"satp_layer=ensure_server,ms={tEnsure1 - tEnsure0},status=ok"
  let snap ← saveState
  let t0 ← IO.monoMsNow
  -- Outer try/catch is defensive: tryInferLayer already classifies all
  -- paths it owns. An exception escaping here would be a Lean-level
  -- tactic crash outside evalReturnedTactic — extremely rare, but we
  -- still tag it as `exec` rather than silently swallowing.
  let result : Except SatpFailKind Unit ← try
    tryInferLayer stxRef cfg #[] false 40 traceScript "policy" #[] false
  catch e => pure (.error (.exec e.toMessageData))
  let t1 ← IO.monoMsNow
  let elapsed := t1 - t0
  match result with
  | .ok () =>
      logInfoAt stxRef s!"satp_layer=policy,ms={elapsed},status=ok"
      return
  | .error kind =>
      snap.restore
      logInfoAt stxRef s!"satp_layer=policy,ms={elapsed},status=fail,fail_kind={kind.label}"
      if traceScript then
        throwError "satp?: policy failed ({kind.label}): {kind.reason}"
      else
        fallbackToAesop stxRef m!"satp policy failed ({kind.label}): {kind.reason}"

/--
`satp` (Steering Aesop for Theorem Proving) queries a local SATP inference
service for a tailored `aesop` configuration and runs it inside Lean.

Usage:
- `satp` — single-shot policy call on the current goal state.  The model
  only sees the goal; its built-in retrieval picks the rules.
- `satp [h₁, h₂, …]` — same call; the bracket list is accepted for
  API back-compat but **ignored** (the policy model does not ingest
  user hints and we no longer inject them post-inference as aesop
  rules).  To re-enable hint injection, reconnect `lemmaNames` inside
  `runSatpCascade`.

If the Python service is unavailable or the policy call fails, `satp`
logs one warning and falls back to plain `aesop` so usage in the
elaboration of ordinary proofs is robust to transient server issues.
The server endpoint is `127.0.0.1:5177` by default; override with
`SATP_SERVER_HOST` / `SATP_SERVER_PORT` / `SATP_REQUEST_TIMEOUT` env vars.
-/
syntax (name := satp) "satp" (ppSpace "[" (term),* "]")? : tactic

/--
`satp?` is the evaluation-oriented variant: it prints the exact tactic
the policy ran as a "Try this" suggestion (for distill-back into a
concrete proof) AND throws on any failure (server unavailable or
policy miss) instead of falling back to `aesop`. The no-fallback
semantics keep `satp` mode's closure count attributable to the 0.2B
policy alone, so `satp - aesop_plain` cleanly measures the learned
policy's marginal value. Use bare `satp` (no `?`) for ordinary proof
scripts that want robust behavior on transient failures.
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
