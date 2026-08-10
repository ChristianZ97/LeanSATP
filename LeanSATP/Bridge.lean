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
  return (((← defaultRepoRoot) / "cache_v427").normalize.toString)

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
Render the current tactic state with Lean's standard goal printer
(`Meta.ppGoal`) — the same rendering that produced the policy's
training / eval `goal_state` inputs (same-type binders merged into one
line: `b h v : ℝ`). The byt5 policy is byte-sensitive, so any rendering
drift shifts decodes: the previous hand-rolled one-hypothesis-per-line
renderer perturbed the policy input on essentially every problem.
2026-07-10 probe: under `ppGoal`, 182/208 parseable minif2f-test
statements render byte-equal to the dataset `goal_state`; the remaining
26 are stale-era pretty-print artifacts no current printer reproduces.
-/
private def collectGoalState : TacticM String := withMainContext do
  return (← ppGoal (← getMainGoal)).pretty

/--
The satp policy is trained and reproduced with goal-state policy inputs.
Theorem-shaped proofs are elaborated by Lean first; the policy always
sees the resulting tactic state.

Codex pt.8 Finding 2 (retry alternate mode before fail) intentionally
NOT implemented: the retry would double HTTP latency for every failed
policy call. If a future ablation needs it, wire a second `tryInferLayer`
call between `snap.restore` and the trailing `throwError` in
`runSatpCascade`.
-/
private def collectSatpInput : TacticM String :=
  collectGoalState

private def elabUserLemmaNames (terms : Array (TSyntax `term)) : TacticM (Array String) := do
  let mut names := #[]
  for term in terms do
    match term with
    | `(term| $id:ident) =>
      names := names.push id.getId.toString
    | _ =>
      throwError "satp only supports identifier lemmas in [ ... ], got: {term}"
  -- Codex pt.8 Finding 1 remediation: surface the discard so callers
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
    (policyInput : String)
    (userLemmas : Array String)
    (stripRetrieval : Bool := false)
    (hintPriority : Nat := 40) : TacticM (Except MessageData String) := do
  -- `formal_statement` is the legacy JSON field name. In v2 it carries
  -- the policy input, which is a Lean goal-state string.
  -- Extra fields on top of the legacy (formal_statement / user_lemmas /
  -- tactic_name) contract drive the satp? 3-layer cascade:
  --   • strip_retrieval = true  → server drops the policy's retrieval-
  --     sourced lemma rules (Layer 1 "trust sketch" branch)
  --   • hint_priority          → priority_pct used when appending
  --     user_lemmas as aesop rules (50 for cascade layers that carry
  --     hints, 40 for the legacy code path)
  let requestBody := Json.compress <| Json.mkObj [
    ("formal_statement", Json.str policyInput),
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

-- Runtime-inclusive except-shaping. Plain `try/catch` — and the outer
-- `first` combinator at every call site — RETHROW runtime exceptions
-- (`maxRecDepth`/heartbeats, see `Core.tryCatch`), so e.g. a `by simp`
-- rule inside the emitted config blowing `maxRecDepth` would abort the
-- whole gap and shadow every later cascade branch. Interrupts still
-- propagate (tryCatchRuntimeEx never catches those).
private def exceptRuntime (x : TacticM α) : TacticM (Except MessageData α) :=
  tryCatchRuntimeEx (Except.ok <$> x) fun e => do
    -- Heartbeats are monotonic and survive backtracking — containing an
    -- exhaustion would just re-time-out downstream (or let a cheap
    -- fallback slip in under an already-blown budget). Rethrow those;
    -- contain depth blowups + regular errors only.
    if e.isMaxHeartbeat then throw e else pure (.error e.toMessageData)

private def evalReturnedTactic
    (tacticSeq : TSyntax ``tacticSeq) : TacticM (Except MessageData Unit) :=
  exceptRuntime (evalTactic tacticSeq)

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
  | input    (msg : MessageData)   -- could not collect policy input
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
  let policyInputE ← exceptRuntime collectSatpInput
  let policyInput ←
    match policyInputE with
    | .ok pi => pure pi
    | .error msg => return .error (.input msg)
  match ← callInferenceService cfg policyInput userLemmas stripRetrieval hintPriority with
  | .error reason => return .error (.request reason)
  | .ok tacticString =>
      match ← parseReturnedTactic tacticString with
      | .error reason => return .error (.parse reason)
      | .ok tacticSeq =>
          -- Audit only goals that are ours to close: goals already
          -- assigned before the policy ran (possible in arbitrary user
          -- tactic states) must not trip the sorryAx guard below.
          let gs ← (← getGoals).filterM fun g => return !(← g.isAssigned)
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
                -- sorryAx-closure guard: errToSorry recovery (or an
                -- emitted `sorry`) can "close" a goal with sorryAx — a
                -- fake win that would shadow later cascade branches.
                -- Aesop.hasSorry (not Expr.hasSorry ∘ instantiateMVars):
                -- it also follows delayed mvar assignments.
                for g in gs do
                  if ← Aesop.hasSorry (mkMVar g) then
                    return .error (.exec m!"emitted tactic closed the goal with sorryAx")
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
  -- Emit before any saveState region so the record survives
  -- `snap.restore` on policy failure.
  logInfoAt stxRef "satp_input_mode=goal"
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
  -- Message-log watermark (2026-08-10): term-elab failures inside the
  -- emitted config can be LOGGED + errToSorry'd instead of thrown —
  -- tryInferLayer then reports success while a file-level error sits in
  -- the log (⇒ lake rc=1), and no restore ever runs on that
  -- spurious-success path (only failure paths restore, and
  -- `Tactic.SavedState.restore` does reset the log). So: any new
  -- error-severity message during the policy eval ⇒ downgrade to
  -- exec-failure and let the failure path's restore drop the messages.
  let msgsSaved ← Core.getMessageLog
  let errCount : MessageLog → Nat := fun l =>
    (l.toList.filter (fun m => m.severity matches .error)).length
  let t0 ← IO.monoMsNow
  -- Outer catch is defensive: tryInferLayer already classifies all
  -- paths it owns. Runtime-inclusive (exceptRuntime) so a stray
  -- maxRecDepth outside evalReturnedTactic still fails the layer
  -- instead of aborting the gap.
  let result : Except SatpFailKind Unit ←
    match ← exceptRuntime
      (tryInferLayer stxRef cfg #[] false 40 traceScript "policy" #[] false) with
    | .ok inner => pure inner
    | .error msg => pure (.error (.exec msg))
  let t1 ← IO.monoMsNow
  let elapsed := t1 - t0
  let result : Except SatpFailKind Unit ←
    match result with
    | .ok () =>
        if errCount (← Core.getMessageLog) > errCount msgsSaved then
          pure (.error (.exec m!"emitted tactic logged error diagnostics (logged-error guard)"))
        else
          pure (.ok ())
    | e => pure e
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
