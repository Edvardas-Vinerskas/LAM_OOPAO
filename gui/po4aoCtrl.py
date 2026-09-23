#!/usr/bin/env python3
"""GUI for the PO4AO reinforcement-learning controller (po4ao_rama.py).

Run it in the rtc environment:  conda activate rtc && python po4aoCtrl.py

  - edit the parameters of po4ao_config.py and save it (comments kept,
    timestamped backup), then START / STOP the script in the PO4AO_RL tmux
    session; the script reads the config once, at launch (white labels);
  - once the script runs it publishes the control SHMs of po4ao_interface.py
    and the "Live control" group is enabled: loop mode (open / integrator /
    policy), training and exploration on-off, WARM-UP, checkpoint save / load,
    RELOAD POLICY, APPLY LIVE for the blue parameters, status line;
  - watch the bench prerequisites (CNN modes flowing, NN / linear integrators
    open, RL DM channel) and the script console.

Typical session: START -> WARM-UP (integrator + noise, then pre-training, then
the policy takes over with training on) -> SAVE MODELS / SAVE BUFFERS.
Later: START -> LOAD CHECKPOINT -> CLOSE (policy), no warm-up needed.
"""

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import QRegExp
from PyQt5.QtGui import QRegExpValidator, QTextCursor
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox
from PyQt5.uic import loadUiType

import dao

HERE = os.path.dirname(os.path.abspath(__file__))
RL_DIR = os.path.dirname(HERE)                      # folder holding po4ao_rama.py, e.g. .../evinerskas/PO4AO_gui
SCRIPT = os.path.join(RL_DIR, "po4ao_rama.py")
CONFIG = os.path.join(RL_DIR, "po4ao_config.py")
CONDA_SH = "/home/rama/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV = "rtc"
TMUX_SESSION = "PO4AO_RL"

# Bench SHMs the script depends on
SHM_MODES_NN = "/tmp/pyrModesNN.im.shm"
SHM_LOOP_NN = "/tmp/lpCmdNN.im.shm"
SHM_LOOP_LIN = "/tmp/lpCmd.im.shm"
SHM_RL_DM = "/tmp/dm1Cmd05.im.shm"


# # Control SHMs of the threaded script (PROPOSAL.md). Absent today.
# SHM_CMD = "/tmp/po4aoCmd.im.shm"        # int32 (8,1)
# SHM_PARAMS = "/tmp/po4aoParams.im.shm"  # float32 (len(LIVE_KEYS),1)
# SHM_STATUS = "/tmp/po4aoStatus.im.shm"  # float32 (16,1)
# CMD_LOOP, CMD_TRAIN, CMD_EXPLORE, CMD_SAVE_MODELS, CMD_SAVE_BUFFERS, CMD_RELOAD_POLICY = range(6)
# LOOP_OPEN, LOOP_INTEGRATOR, LOOP_POLICY = 0, 1, 2
# STATUS_NAMES = ("iteration", "step", "loop Hz", "reward", "dyn loss", "pol loss",
#                 "training", "policy version")


#TODO what was the point of this change to po4ao_interface instead of using the commented out code above?
# Control SHM protocol, shared with po4ao_rama.py (same folder as the script)
sys.path.insert(0, RL_DIR)
from po4ao_interface import (  # noqa: E402
    SHM_CMD, SHM_PARAMS, SHM_STATUS, CHECKPOINT_REQUEST_FILE,
    CMD_LOOP, CMD_TRAIN, CMD_EXPLORE, CMD_SAVE_MODELS, CMD_SAVE_BUFFERS, CMD_RELOAD_POLICY,
    CMD_LOAD_CHECKPOINT, CMD_WARMUP, CMD_QUIT, LOOP_OPEN, LOOP_INTEGRATOR, LOOP_POLICY, LOOP_NAMES,
    LIVE_PARAMS, STATUS_NAMES, ST_TRAINING, ST_MODE, ST_EXPLORE, ST_STATE, ST_POLICY_READY,
    STATE_NAMES)

DATA_SHAPE = 11  # DM grid size (MDP.data_shape, fixed by dm1Map); for the buffer memory estimate

STYLE = """
    QMainWindow, QWidget { background-color: #1e1e1e; color: #cccccc; }
    QPushButton {
        background-color: #3a3a3a; color: #cccccc;
        border: 1px solid #555; padding: 4px 8px; border-radius: 3px;
    }
    QPushButton:hover { background-color: #4a4a4a; }
    QPushButton:pressed, QPushButton:checked { background-color: #555; }
    QPushButton:disabled { background-color: #2a2a2a; color: #666666; }
    QCheckBox, QLabel { color: #cccccc; }
    QCheckBox::indicator { width: 14px; height: 14px; }
    QCheckBox::indicator:unchecked { background-color: #aaaaaa; border: 1px solid #ccc; }
    QCheckBox:disabled { color: #666666; }
    QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit {
        background-color: #3a3a3a; color: #cccccc; border: 1px solid #555;
    }
    QDoubleSpinBox:disabled, QSpinBox:disabled { color: #777777; }
    QGroupBox {
        border: 1px solid #2dd4bf; border-radius: 8px;
        margin-top: 10px; padding-top: 6px;
    }
    QGroupBox::title {
        subcontrol-origin: margin; left: 10px; padding: 0 4px;
        color: #2dd4bf; font-weight: bold;
    }
"""

# --- PARAMETER TABLE ---
SECTIONS = {
    "RL": "Reinforcement learning",
    "MDP": "MDP",
    "replay_buffers": "Replay buffers",
    "integrator": "Integrator",
    "save_and_load": "Save / load",
}

# live=True: sent to the running script with APPLY LIVE (blue label). The set and
# order must match LIVE_PARAMS in po4ao_interface.py (checked below).
PARAMS = [
    dict(section="RL", key="episode_length", type="int", min=100, max=1000000, step=1000,
         tip="Frames per episode (replay buffers are sized with it)"),
    dict(section="RL", key="warmup_episodes", type="int", min=0, max=1000,
         tip="Integrator (+ noise) episodes filling the warm-up buffer"),
    dict(section="RL", key="max_sigma", type="float", min=0, max=1, decimals=4, step=0.001, live=True,
         tip="Initial exploration noise (DM units, x1e-2 in the script)"),
    dict(section="RL", key="loss_function_penalty", label="action penalty", type="float",
         min=0, max=10, decimals=3, step=0.05, live=True,
         tip="loss = mean(state^2) + penalty * mean(action^2)"),
    #TODO do I need these?
    dict(section="RL", key="policy_after_warmup", label="policy after warm-up", type="bool",
         tip="When pre-training ends: switch to the policy with training on (else stay on the integrator)"),
    dict(section="RL", key="auto_policy_update", label="auto policy update", type="bool",
         tip="Swap in every newly trained policy at the next episode (else use RELOAD POLICY)"),

    dict(section="MDP", key="n_history", type="int", min=1, max=512,
         tip="Past frames fed to the networks (changes the network input size)"),
    dict(section="MDP", key="planning_horizon", type="int", min=1, max=64, live=True,
         tip="Dynamics roll-out steps when training the policy"),

    dict(section="replay_buffers", key="replay_size", type="int", min=1, max=1000,
         tip="Episodes kept in the on-line buffer (GPU memory)"),
    dict(section="replay_buffers", key="warmup_memory", type="int", min=1, max=1000,
         tip="Episodes kept in the warm-up buffer (GPU memory)"),
    dict(section="replay_buffers", key="train_warmup_percent", label="warm-up sample fraction",
         type="float", min=0, max=1, decimals=2, step=0.05, live=True,
         tip="Probability a training batch is drawn from the warm-up buffer"),

    dict(section="integrator", key="gain", type="float", min=0, max=2, decimals=3, step=0.01, live=True),
    dict(section="integrator", key="leak", type="float", min=0, max=1, decimals=4, step=0.001, live=True,
         tip="Command memory, also used by the RL step"),
    dict(section="integrator", key="n_modes", type="int", min=1, max=96,
         tip="Modes kept by the KL projection of the policy output"),
    dict(section="integrator", key="offset", label="subtract dm1CmdOffset", type="bool", live=True),
    dict(section="integrator", key="command_clamp", label="|command| clamp", type="float", min=0.01, max=2, decimals=2, step=0.05,
         tip="Limit on the integrated dm1Cmd05 command"),

    dict(section="save_and_load", key="warmup_at_start", label="warm-up at start", type="bool",
         tip="Run the warm-up + pre-training right after launch (old behaviour); else use the WARM-UP button"),
    dict(section="save_and_load", key="save_after_pretrain", label="save after pre-training", type="bool",
         tip="Checkpoint models + buffers in the run directory when pre-training ends"),
    dict(section="save_and_load", key="save_at_exit", label="save at exit", type="bool",
         tip="Checkpoint models + buffers on quit / Ctrl-C"),
]
# --- END PARAMETER TABLE ---

LIVE_KEYS = [(p["section"], p["key"]) for p in PARAMS if p.get("live")]
if LIVE_KEYS != LIVE_PARAMS:
    sys.exit("po4aoCtrl.py: live parameters differ from LIVE_PARAMS in po4ao_interface.py:\n"
             f"  GUI:    {LIVE_KEYS}\n  script: {LIVE_PARAMS}")

Ui_MainWindow, _ = loadUiType(os.path.join(HERE, "po4aoCtrl.ui"))


def read_config(path):
    spec = importlib.util.spec_from_file_location("po4ao_config_gui", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.config


def format_value(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    return repr(float(value))


def write_config(path, values):
    """Replace only the values in po4ao_config.py, keeping layout and comments."""
    text = open(path).read()
    for (section, key), value in values.items():
        sec = re.search(r"'%s'\s*:\s*\{" % re.escape(section), text)
        if sec is None:
            raise ValueError(f"section '{section}' not found in {path}")
        end = text.index("}", sec.end())
        m = re.compile(r"('%s'\s*:\s*)([^,#\n{]+?)(\s*(,|#|\n))" % re.escape(key)).search(text, sec.end(), end)
        if m is None:
            raise ValueError(f"'{section}.{key}' not found in {path}")
        text = text[:m.start(2)] + format_value(value) + text[m.end(2):]
    shutil.copy2(path, path + time.strftime(".bak-%Y%m%dT%H%M%S"))
    open(path, "w").write(text)


def open_shm(path):
    # dao segfaults when asked to open a missing SHM without an array
    if not os.path.exists(path):
        return None
    try:
        return dao.shm(path)
    except Exception:
        return None


def tmux(*args):
    return subprocess.run(["tmux", *args], capture_output=True, text=True)


class Po4aoCtrlApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setStyleSheet(STYLE)

        self.scriptPathLabel.setText(SCRIPT)
        self.condaEnvEdit.setText(CONDA_ENV)
        self.condaEnvEdit.setValidator(QRegExpValidator(QRegExp(r"[A-Za-z0-9_.\-]*"), self.condaEnvEdit))
        self.cpuEdit.setValidator(QRegExpValidator(QRegExp(r"[0-9,\-]*"), self.cpuEdit))
        self.configPathLabel.setText(CONFIG)
        self._widgets = {(p["section"], p["key"]): getattr(self, f"cfg__{p['section']}__{p['key']}")
                         for p in PARAMS}
        self._types = {(p["section"], p["key"]): p["type"] for p in PARAMS}
        self._saved = {}
        self._modes_cnt = None
        self._modes_t = None
        self._modes_hz = 0.0
        self._gpu_t = 0.0

        self.reloadConfigButton.clicked.connect(self.load_config)
        self.saveConfigButton.clicked.connect(self.save_config)
        self.startButton.clicked.connect(self.start_script)
        self.stopButton.clicked.connect(self.stop_script)
        self.logButton.clicked.connect(self.open_log)
        self.closeLoopButton.clicked.connect(lambda: self._send_cmd(CMD_LOOP, LOOP_POLICY))
        self.integratorButton.clicked.connect(lambda: self._send_cmd(CMD_LOOP, LOOP_INTEGRATOR))
        self.openLoopButton.clicked.connect(lambda: self._send_cmd(CMD_LOOP, LOOP_OPEN))
        self.trainingCheckBox.clicked.connect(lambda c: self._send_cmd(CMD_TRAIN, int(c)))
        self.explorationCheckBox.clicked.connect(lambda c: self._send_cmd(CMD_EXPLORE, int(c)))
        self.saveModelsButton.clicked.connect(lambda: self._bump_cmd(CMD_SAVE_MODELS))
        self.saveBuffersButton.clicked.connect(lambda: self._bump_cmd(CMD_SAVE_BUFFERS))
        self.reloadPolicyButton.clicked.connect(lambda: self._bump_cmd(CMD_RELOAD_POLICY))
        self.applyLiveButton.clicked.connect(self.apply_live)
        self.warmupButton.clicked.connect(self.request_warmup)
        self.loadCheckpointButton.clicked.connect(self.request_load_checkpoint)
        self.browseCheckpointButton.clicked.connect(self.browse_checkpoint)
        self.checkpointEdit.setText(os.path.join(RL_DIR, "PO4AO", "logs"))
        for w in self._widgets.values():
            sig = w.toggled if hasattr(w, "toggled") and not hasattr(w, "value") else w.valueChanged
            sig.connect(self._update_budget)

        self.shmModesNN = open_shm(SHM_MODES_NN)
        self.shmLoopNN = open_shm(SHM_LOOP_NN)
        self.shmLoopLin = open_shm(SHM_LOOP_LIN)
        self.shmRlDm = open_shm(SHM_RL_DM)
        self.shmCmd = self.shmParams = self.shmStatus = None

        self.load_config()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(1000)
        self.refresh()

    # ------------------------------------------------------------------ config
    def _get(self, key):
        w = self._widgets[key]
        return bool(w.isChecked()) if self._types[key] == "bool" else w.value()

    def _set(self, key, value):
        w = self._widgets[key]
        w.blockSignals(True)
        if self._types[key] == "bool":
            w.setChecked(bool(value))
        elif self._types[key] == "int":
            w.setValue(int(value))
        else:
            w.setValue(float(value))
        w.blockSignals(False)

    def load_config(self):
        try:
            config = read_config(CONFIG)
        except Exception as e:
            QMessageBox.critical(self, "PO4AO", f"Cannot read {CONFIG}:\n{e}")
            return
        self._saved = {}
        for key in self._widgets:
            section, name = key
            if section in config and name in config[section]:
                self._set(key, config[section][name])
                self._saved[key] = self._get(key)
            else:
                self._widgets[key].setEnabled(False)
        self._update_budget()

    def _dirty(self):
        return {k: self._get(k) for k in self._saved if self._get(k) != self._saved[k]}

    def save_config(self):
        changed = self._dirty()
        if not changed:
            return True
        try:
            write_config(CONFIG, changed)
        except Exception as e:
            QMessageBox.critical(self, "PO4AO", f"Cannot write {CONFIG}:\n{e}")
            return False
        self._saved.update(changed)
        self._update_budget()
        return True

    #TODO what is update_budget?
    def _update_budget(self, *_):
        hz = self._modes_hz if self._modes_hz > 1 else 800.0
        ep = self._get(("RL", "episode_length")) / hz
        warm = ep * self._get(("RL", "warmup_episodes"))
        n = DATA_SHAPE
        episodes = self._get(("replay_buffers", "replay_size")) + self._get(("replay_buffers", "warmup_memory"))
        gib = episodes * self._get(("RL", "episode_length")) * n * n * 4 * 3 / 2**30
        dirty = len(self._dirty())
        self.budgetLabel.setText(
            f"at {hz:.0f} Hz: episode {ep:.1f} s, warm-up {warm / 60:.1f} min"
            f"  |  replay buffers {gib:.2f} GiB GPU"
            + (f"  |  {dirty} unsaved change(s)" if dirty else ""))

    # ------------------------------------------------------------------ run
    def _running(self):
        return subprocess.run(["pgrep", "-f", "po4ao_rama.py"], capture_output=True).returncode == 0

    def start_script(self):
        if self._running():
            return
        if self._dirty():
            if QMessageBox.question(self, "PO4AO", "Save the edited parameters before starting?") \
                    != QMessageBox.Yes or not self.save_config():
                return
        warn = []
        if self.shmLoopNN is not None and float(np.asarray(self.shmLoopNN.get_data()).flat[0]) != 0:
            warn.append("the NN integrator loop is closed (it also drives DM1)")
        if self.shmLoopLin is not None and float(np.asarray(self.shmLoopLin.get_data()).flat[0]) != 0:
            warn.append("the linear integrator loop is closed (it also drives DM1)")
        if self._modes_hz < 1:
            warn.append("pyrModesNN is not updating (CNN not running?): the script will block")
        msg = "Start po4ao_rama.py? It writes DM1 commands on dm1Cmd05."
        if warn:
            msg += "\n\nWarning:\n - " + "\n - ".join(warn)
        if QMessageBox.question(self, "PO4AO", msg) != QMessageBox.Yes:
            return
        cpus = self.cpuEdit.text().strip()
        env = self.condaEnvEdit.text().strip() or CONDA_ENV
        pin = f"taskset -c {cpus} " if cpus else ""
        # Same as by hand: conda activate <env>, then python from that env.
        # po4ao_config is imported from the script folder, but PO4AO/logs is
        # relative to the working directory: run from the script folder.
        command = (f"source {CONDA_SH} && conda activate {env} && cd {RL_DIR} && "
                   f"{pin}python -u {os.path.basename(SCRIPT)}; "
                   "echo '[po4ao_rama.py exited]'; exec bash")
        self.runCommandLabel.setText(f"conda activate {env} && {pin}python -u {SCRIPT}")
        if tmux("has-session", "-t", TMUX_SESSION).returncode == 0:
            tmux("send-keys", "-t", TMUX_SESSION, command, "C-m")
        else:
            tmux("new-session", "-d", "-s", TMUX_SESSION, "/bin/bash", "-lc", command)
        self.refresh()

    def stop_script(self):
        if not self._running():
            return
        if QMessageBox.question(self, "PO4AO", "Stop po4ao_rama.py?\n"
                                "It flattens dm1Cmd05 and, if 'save at exit' is set, "
                                "writes a checkpoint first.") != QMessageBox.Yes:
            return
        if self.shmCmd is not None:
            self._bump_cmd(CMD_QUIT)         # clean shutdown through the control SHM
        else:
            tmux("send-keys", "-t", TMUX_SESSION, "C-c", "")

    def open_log(self):
        if tmux("has-session", "-t", TMUX_SESSION).returncode != 0:
            return
        subprocess.Popen(["xterm", "-bg", "black", "-fg", "#cccccc", "-geometry", "160x50",
                          "-T", TMUX_SESSION, "-e", "tmux", "attach", "-t", TMUX_SESSION])

    # ------------------------------------------------------------------ live control
    def _attach_control(self, running):
        # handles are dropped whenever the script is down and re-opened when it runs
        if not running or not os.path.exists(SHM_CMD):
            self.shmCmd = self.shmParams = self.shmStatus = None
        elif self.shmCmd is None:
            self.shmCmd = open_shm(SHM_CMD)
            self.shmParams = open_shm(SHM_PARAMS)
            self.shmStatus = open_shm(SHM_STATUS)
        available = self.shmCmd is not None
        for w in (self.closeLoopButton, self.integratorButton, self.openLoopButton,
                  self.trainingCheckBox, self.explorationCheckBox, self.saveModelsButton,
                  self.saveBuffersButton, self.reloadPolicyButton, self.warmupButton,
                  self.loadCheckpointButton):
            w.setEnabled(available)
        self.applyLiveButton.setEnabled(available and self.shmParams is not None)
        return available

    def request_warmup(self):
        n = self._saved.get(("RL", "warmup_episodes"), "?")
        if QMessageBox.question(self, "PO4AO", f"Run the warm-up now?\n{n} integrator episodes with "
                                "exploration noise, then pre-training. Any warm-up buffer in memory is "
                                "cleared.") != QMessageBox.Yes:
            return
        self._bump_cmd(CMD_WARMUP)

    def browse_checkpoint(self):
        start = self.checkpointEdit.text().strip() or RL_DIR
        path = QFileDialog.getExistingDirectory(self, "Checkpoint directory", start)
        if path:
            self.checkpointEdit.setText(path)

    def request_load_checkpoint(self):
        path = self.checkpointEdit.text().strip()
        files = [f for f in ("models.pt", "replay.pt", "replay_warmup.pt") if os.path.exists(os.path.join(path, f))]
        if not files:
            QMessageBox.warning(self, "PO4AO", f"No models.pt / replay*.pt in\n{path}")
            return
        if QMessageBox.question(self, "PO4AO", f"Load {', '.join(files)} from\n{path}?\n\n"
                                "The loop opens (DM flat) while loading and stays open.") != QMessageBox.Yes:
            return
        try:
            with open(CHECKPOINT_REQUEST_FILE, "w") as f:
                f.write(path)
        except OSError as e:
            QMessageBox.critical(self, "PO4AO", f"Cannot write {CHECKPOINT_REQUEST_FILE}:\n{e}")
            return
        self._bump_cmd(CMD_LOAD_CHECKPOINT)

    def _send_cmd(self, index, value):
        if self.shmCmd is None:
            return
        cmd = np.asarray(self.shmCmd.get_data()).copy()
        cmd.flat[index] = value
        self.shmCmd.set_data(cmd)

    def _bump_cmd(self, index):
        if self.shmCmd is None:
            return
        cmd = np.asarray(self.shmCmd.get_data()).copy()
        cmd.flat[index] += 1
        self.shmCmd.set_data(cmd)

    def apply_live(self):
        if self.shmParams is None:
            return
        vals = np.asarray(self.shmParams.get_data()).copy()
        for i, key in enumerate(LIVE_KEYS[:vals.size]):
            vals.flat[i] = float(self._get(key))
        self.shmParams.set_data(vals)

    # ------------------------------------------------------------------ periodic refresh
    @staticmethod
    def _scalar(shm):
        try:
            return float(np.asarray(shm.get_data()).flat[0])
        except Exception:
            return None

    def _set_state(self, label, text, ok):
        label.setText(text)
        label.setStyleSheet("color: #44dd44;" if ok else "color: #ff4444; font-weight: bold;")

    #TODO one more function for me to figure out
    def _show_status(self, live):
        """po4aoStatus -> one summary line; mirror the actual mode / flags into the controls."""
        mode_buttons = {LOOP_OPEN: self.openLoopButton, LOOP_INTEGRATOR: self.integratorButton,
                        LOOP_POLICY: self.closeLoopButton}
        if not (live and self.shmStatus is not None):
            self.liveStateLabel.setText("Live control: not available (script not running)")
            for b in mode_buttons.values():
                b.setStyleSheet("")
            return
        st = np.asarray(self.shmStatus.get_data()).ravel()
        s = dict(zip(STATUS_NAMES, st.tolist()))
        mode, state = int(st[ST_MODE]), int(st[ST_STATE])
        training, explore, ready = st[ST_TRAINING] > 0.5, st[ST_EXPLORE] > 0.5, st[ST_POLICY_READY] > 0.5
        noise = f" +noise sigma {s['sigma']:.4g}" if explore else ""
        policy = f"v{s['policy version']:.0f}" if ready else "none"
        self.liveStateLabel.setText(
            f"{STATE_NAMES[state]} | {LOOP_NAMES[mode].upper()}{noise} | training {'on' if training else 'off'}"
            f" | policy {policy}"
            f" | ep {s['episode']:.0f} @ {s['loop Hz']:.0f} Hz reward {s['reward']:.4g}"
            f" | dyn {1e3 * s['dyn loss']:.4g} pol {1e3 * s['pol loss']:.4g} (x1e-3)"
            f" | replay {s['replay episodes']:.0f} warm-up {s['warm-up episodes']:.0f} rounds {s['trainer rounds']:.0f}")
        for m, b in mode_buttons.items():
            b.setStyleSheet("border: 2px solid #2dd4bf; color: #2dd4bf;" if m == mode else "")
        self.closeLoopButton.setEnabled(ready)
        for cb, val in ((self.trainingCheckBox, training), (self.explorationCheckBox, explore)):
            cb.blockSignals(True)
            cb.setChecked(bool(val))
            cb.blockSignals(False)

    def refresh(self):
        running = self._running()
        session = tmux("has-session", "-t", TMUX_SESSION).returncode == 0
        self._set_state(self.runStateLabel, "RUNNING" if running else
                        ("stopped (session kept)" if session else "stopped"), running)
        self.runStateLabel.setStyleSheet(
            "color: #ff4444; font-weight: bold;" if running else "color: #44dd44; font-weight: bold;")
        self.startButton.setEnabled(not running)
        self.stopButton.setEnabled(running)
        self.logButton.setEnabled(session)

        live = self._attach_control(running)
        #TODO what is this doing?
        self._show_status(live)

        # prerequisites
        if self.shmModesNN is not None:
            cnt, now = self.shmModesNN.get_counter(), time.time()
            if self._modes_cnt is not None:
                self._modes_hz = (cnt - self._modes_cnt) / max(now - self._modes_t, 1e-3)
            self._modes_cnt, self._modes_t = cnt, now
            self._set_state(self.preCnnLabel, f"{self._modes_hz:.0f} Hz", self._modes_hz > 1)
        else:
            self._set_state(self.preCnnLabel, "SHM missing (RAMA not started?)", False)
        for shm, lbl in ((self.shmLoopNN, self.preNnLoopLabel), (self.shmLoopLin, self.preLinLoopLabel)):
            v = self._scalar(shm) if shm is not None else None
            if v is None:
                self._set_state(lbl, "SHM missing", False)
            else:
                self._set_state(lbl, "open" if v == 0 else "CLOSED (also drives DM1)", v == 0)
        if self.shmRlDm is not None:
            peak = float(np.max(np.abs(np.asarray(self.shmRlDm.get_data()))))
            self.preDmLabel.setText(f"max |cmd| = {peak:.4f}")
            self.preDmLabel.setStyleSheet("color: #cccccc;")
        else:
            self._set_state(self.preDmLabel, "SHM missing", False)
        if time.time() - self._gpu_t > 5:
            self._gpu_t = time.time()
            q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True)
            try:
                used, total, util = (float(x) for x in q.stdout.splitlines()[0].split(","))
                self.preGpuLabel.setText(f"{used / 1024:.1f} / {total / 1024:.0f} GiB, {util:.0f} %")
            except Exception:
                self.preGpuLabel.setText("nvidia-smi unavailable")
        self._update_budget()

        # console: the script logs once per episode, so the pane stays useful when live
        if session:
            # what the console shows (history included, wrapped lines joined);
            # follow the end unless the user scrolled up to read
            pane = tmux("capture-pane", "-p", "-J", "-t", TMUX_SESSION, "-S", "-2000").stdout.rstrip()
            if pane != self.logText.toPlainText():
                bar = self.logText.verticalScrollBar()
                at_end = bar.value() >= bar.maximum() - 2
                keep = bar.value()
                self.logText.setPlainText(pane)
                if at_end:
                    self.logText.moveCursor(QTextCursor.End)
                    self.logText.ensureCursorVisible()
                else:
                    bar.setValue(keep)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Po4aoCtrlApp()
    window.show()
    sys.exit(app.exec_())
