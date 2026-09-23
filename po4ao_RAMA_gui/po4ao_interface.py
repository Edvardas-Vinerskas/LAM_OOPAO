"""SHM protocol between po4ao_rama.py (controller) and po4aoCtrl.py (GUI).

Kept free of torch / Qt so both sides can import it. Any script can drive the
controller the same way: write po4aoCmd, read po4aoStatus.
"""

import os

import dao

# These three SHMs must exist on the bench before the controller starts (they are
# not created by this code); the controller opens, zeroes and writes them.
SHM_CMD = "/tmp/po4aoCmd.im.shm"          # int32 (N_CMD, 1), written by the GUI
SHM_PARAMS = "/tmp/po4aoParams.im.shm"    # float32 (len(LIVE_PARAMS), 1), written by the GUI
SHM_STATUS = "/tmp/po4aoStatus.im.shm"    # float32 (N_STATUS, 1), written by the controller
CHECKPOINT_REQUEST_FILE = "/home/rama/rama-dev/evinerskas/PO4AO_gui/po4aoCheckpoint.txt"  # directory to load, written before CMD_LOAD_CHECKPOINT

# --- po4aoCmd -------------------------------------------------------------
# Slots 0-2 are settings (the controller applies the value); the others are
# request counters: the writer increments, the controller acts on the change.
(CMD_LOOP, CMD_TRAIN, CMD_EXPLORE,
 CMD_SAVE_MODELS, CMD_SAVE_BUFFERS, CMD_RELOAD_POLICY,
 CMD_LOAD_CHECKPOINT, CMD_WARMUP, CMD_QUIT) = range(9)
CMD_SETTINGS = (CMD_LOOP, CMD_TRAIN, CMD_EXPLORE)

N_CMD = 16

LOOP_OPEN, LOOP_INTEGRATOR, LOOP_POLICY = 0, 1, 2
LOOP_NAMES = ("open", "integrator", "policy")

# --- po4aoParams ----------------------------------------------------------
# Parameters that can change while the controller runs, in SHM order.
# po4aoCtrl.py marks the same keys live=True and checks the order at start-up.
LIVE_PARAMS = [
    ("RL", "max_sigma"),
    ("RL", "loss_function_penalty"),
    ("MDP", "planning_horizon"),
    ("replay_buffers", "train_warmup_percent"),
    ("integrator", "gain"),
    ("integrator", "leak"),
    ("integrator", "offset"),
]

# --- po4aoStatus ----------------------------------------------------------
(ST_EPISODE, ST_STEP, ST_HZ, ST_REWARD, ST_DYN_LOSS, ST_POL_LOSS,
 ST_TRAINING, ST_VERSION, ST_MODE, ST_EXPLORE, ST_SIGMA, ST_STATE,
 ST_REPLAY_EP, ST_WARMUP_EP, ST_ROUNDS, ST_POLICY_READY) = range(16)

N_STATUS = 16
STATUS_NAMES = ("episode", "step", "loop Hz", "reward", "dyn loss", "pol loss",
                "training", "policy version", "mode", "exploration", "sigma", "state",
                "replay episodes", "warm-up episodes", "trainer rounds", "policy ready")

STATE_IDLE, STATE_WARMUP, STATE_PRETRAIN, STATE_RUN = range(4)
STATE_NAMES = ("idle", "warm-up", "pre-training", "run")



def open_shm(path):
    """Open an existing SHM, None if absent (dao segfaults on a missing file)."""
    if not os.path.exists(path):
        return None
    try:
        return dao.shm(path)
    except Exception:
        return None

