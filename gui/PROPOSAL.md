# PO4AO on RAMA — proposal: from a monolithic script to a controllable service

Status: migration steps 1-4 are implemented inside `po4ao_rama.py` (resident
controller + continuous trainer process, no file split yet); the SHM protocol
lives in `po4ao_interface.py` and is imported by both sides. Differences from
the text below: `po4aoCmd` is int32 (16,1) with extra slots 6 load-checkpoint,
7 warm-up, 8 quit (the checkpoint path goes through `/tmp/po4aoCheckpoint.txt`);
the replay buffer keeps episode alignment with commit / abort instead of a
valid-start mask (OPEN drops the running episode, other mode changes wait for
the episode boundary); the live parameter list is `LIVE_PARAMS` in the
interface module. Untested on the bench so far.

## Why the current script cannot be driven live

- The config is imported once (`from po4ao_config import config`) and copied into
  module globals (`gain`, `leak`, `batch_size`...); nothing re-reads it.
- The main loop is a fixed sequence (warm-up → pre-training → N iterations); there is
  no state you can switch (open / integrator / policy), no pause, no stop other than Ctrl-C.
- Training is started per iteration through `start_q` / `finished_q`: it cannot be
  disabled, and its parameters are frozen at launch.
- The control loop prints at every frame (`print(..., end='\r')`), which costs latency
  at ~800–950 Hz and is the only status channel.

## Target architecture

Three roles, two processes:

```
 ┌──────────────── process 1 (pinned CPUs, e.g. taskset -c 10-20) ─────────────────┐
 │                                                                                  │
 │  Control thread (hot loop, highest priority, no prints, no allocation)           │
 │    wait pyrModesNN → build obs → action by MODE → clamp/leak → dm1Cmd05          │
 │    → append to replay (GPU ring buffer) → update counters                        │
 │    reads Params snapshot only when po4aoCmd / po4aoParams counters change        │
 │                                                                                  │
 │  Supervisor thread (low rate, ~5–10 Hz)                                          │
 │    state machine, episode bookkeeping, sigma schedule, checkpoint requests,      │
 │    policy hand-over from the trainer, watchdog, po4aoStatus publication, logs    │
 └──────────────────────────────────────────────────────────────────────────────────┘
            │ CUDA IPC: replay buffer tensors, published policy weights, version
 ┌──────── process 2 (spawn, other CPUs) ─────────┐
 │  Trainer loop                                   │
 │    while enabled: train dynamics, train policy  │
 │    (live grad steps / batch / horizon / penalty │
 │     / warm-up fraction), publish policy + ver   │
 └─────────────────────────────────────────────────┘
```

Why this split:

- The GIL: training in a thread would stall the control loop. Training stays a separate
  process (as today); the supervisor is a thread because it mostly sleeps and must share
  the controller's objects cheaply.
- One owner per resource: only the control thread writes `dm1Cmd05` and appends to the
  replay; only the trainer writes the trainable networks; the supervisor only
  requests, swaps and reports.

### State machine (supervisor)

`IDLE → WARMUP → PRETRAIN → RUN`, with `RUN` split by loop mode, plus `OPEN` reachable
from anywhere:

| Mode         | Control thread output                         |
|--------------|-----------------------------------------------|
| `OPEN`       | dm1Cmd05 = 0, prev_commands reset, no replay append |
| `INTEGRATOR` | gain·obs (+ exploration noise if enabled)     |
| `POLICY`     | inference policy (+ noise if enabled)         |

Warm-up is just `INTEGRATOR` + exploration with a sigma schedule, so it no longer needs
its own code path. Iteration count becomes a stop condition, not the program structure.

### Policy hand-over

The trainer never touches the inference network. After a training round it copies its
weights into a *publish* buffer (shared CUDA tensors) under an `mp.Lock` and increments a
shared version. The supervisor sees the new version and, between two control steps,
`copy_()`s it into the inference policy (small CNN: well under a ms), or only on the
`RELOAD POLICY` request if auto-swap is off. No half-written weights, no queue pickling.

### Replay buffer

Keep `EfficientExperienceReplay` on the GPU, allocated by process 1 and shared with the
trainer through CUDA IPC (as today via the queue). Add a shared committed write index
(`mp.Value`): the trainer only samples indices below it, so writer and reader need no lock.
`sample_contiguous` currently assumes episode-aligned data; replace it with a "valid start"
mask (no window crossing a mode change or an OPEN gap).

## Control interface (DAO SHMs, same pattern as the RTC)

Change detection on the SHM counter, so the hot loop pays one integer compare per frame.

| SHM                      | Type / shape    | Content |
|--------------------------|-----------------|---------|
| `/tmp/po4aoCmd.im.shm`   | int32 (8,1)     | 0 loop mode (0 open, 1 integrator, 2 policy) · 1 training on/off · 2 exploration on/off · 3 save-models request (increment) · 4 save-buffers request · 5 reload-policy request · 6–7 spare |
| `/tmp/po4aoParams.im.shm`| float32 (N,1)   | live parameters, fixed order = `LIVE_KEYS` in `po4aoCtrl.py` |
| `/tmp/po4aoStatus.im.shm`| float32 (16,1)  | iteration, step, loop Hz, reward, dynamics loss, policy loss, training running, policy version, spare |

Live parameters (blue in the GUI): `max_sigma`, `min_sigma`, `loss_function_penalty`,
`dynamics_grad_steps`, `policy_grad_steps`, `planning_horizon`, `train_warmup_percent`,
`gain`, `leak`, `integrator`, `offset`, `training_batch`.

Restart-only (they size tensors or networks): `n_history`, `n_modes`, `data_shape`,
`filters_per_layer`, `replay_size`, `warmup_memory`, `episode_length`,
`initial_std/mean`, load/save flags.

Why SHMs rather than a socket / JSON file: the rest of RAMA is driven this way, any script
(`campaign/`, notebooks) can drive the RL like the GUI, and reading a counter costs nothing
in the hot loop. A JSON snapshot of the full config is still written per run for the record.

## Safety

- `OPEN` zeroes `dm1Cmd05` and the integrated command, from any state.
- Watchdog in the supervisor: no new `pyrModesNN` for N ms → `OPEN`; reward above a
  threshold (the commented `reward_sum > 34000` guard) → fall back to `INTEGRATOR`.
- Refuse `POLICY` / `INTEGRATOR` while `lpCmdNN` or `lpCmd` is closed (two controllers on
  DM1); the GUI already warns.
- Clamp limits (±0.5 command, ±0.03 action) become explicit parameters, not literals.

## Code layout

```
po4ao/
  config.py      dataclass config + JSON load/save (po4ao_config.py kept as defaults)
  interface.py   SHM protocol shared with the GUI (indices, names, open/create)
  control.py     RTC step, loop modes, noise, replay append (control thread)
  supervisor.py  state machine, hand-over, watchdog, status, checkpoints
  trainer.py     trainer process (train_dynamics / train_policy, publish)
  replay.py      EfficientExperienceReplay + committed index + valid-start mask
  models.py      po4ao_models_upd.py, unchanged
  main.py        wiring, CPU pinning, run directory
```

## Migration, each step testable on the bench

1. Extract `step()` and the episode loops into a `Controller` class that reads a `Params`
   object instead of globals. Same behaviour, same outputs.
2. Add `interface.py`: publish `po4aoStatus`, read `po4aoCmd` (mode, stop), remove the
   per-frame prints. The GUI switches from the tmux log to the status SHM.
3. Turn `start_q` / `finished_q` into the continuous trainer with an enable flag and the
   versioned policy publish.
4. Add the supervisor state machine, live parameters, save / reload requests, watchdog.
5. Run directory under `$DAODATA/po4ao/<timestamp>_<name>` instead of the hard-coded
   relative `PO4AO/logs/<fixed name>` (overwritten at every run today).

Open questions: second GPU (or CUDA MPS) for the trainer to remove inference jitter;
which CPUs for process 2; whether mode changes must wait for an episode boundary.
