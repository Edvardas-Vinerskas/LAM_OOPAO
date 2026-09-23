#!/usr/bin/env python3
"""Generate po4aoCtrl.ui (Qt Designer XML) from the parameter table in po4aoCtrl.py."""
import sys
import importlib.util
from xml.sax.saxutils import escape

gui_py = sys.argv[1]
out = sys.argv[2]
spec = importlib.util.spec_from_file_location("po4aoCtrl_params", gui_py)
mod = importlib.util.module_from_spec(spec)
sys.modules["po4aoCtrl_params"] = mod
# only the tables are needed: exec the module top without Qt / dao
src = open(gui_py).read()
table_src = src[src.index("# --- PARAMETER TABLE ---"):src.index("# --- END PARAMETER TABLE ---")]
ns = {}
exec(table_src, ns)
SECTIONS, PARAMS = ns["SECTIONS"], ns["PARAMS"]

_id = [0]


def prop(name, inner):
    return f'<property name="{name}">{inner}</property>'


def string(s):
    return f"<string>{escape(s)}</string>"


def widget(cls, name, props="", children=""):
    return f'<widget class="{cls}" name="{name}">{props}{children}</widget>'


def item(row, col, content, colspan=1):
    span = f' colspan="{colspan}"' if colspan > 1 else ""
    return f'<item row="{row}" column="{col}"{span}>{content}</item>'


def label(name, text, tooltip="", style=""):
    p = prop("text", string(text))
    if tooltip:
        p += prop("toolTip", string(tooltip))
    if style:
        p += prop("styleSheet", string(style))
    return widget("QLabel", name, p)


def grid(name, items):
    return (f'<layout class="QGridLayout" name="{name}">'
            + prop("horizontalSpacing", "<number>8</number>")
            + prop("verticalSpacing", "<number>4</number>")
            + "".join(items) + "</layout>")


def group(name, title, layout_xml):
    return widget("QGroupBox", name, prop("title", string(title)), layout_xml)


def param_group(section):
    rows = []
    r = 0
    for p in PARAMS:
        if p["section"] != section:
            continue
        key, typ = p["key"], p["type"]
        wname = f"cfg__{section}__{key}"
        tip = p.get("tip", "")
        tip += ("\n" if tip else "") + ("Live: sent to the running script with APPLY LIVE"
                                         if p.get("live") else "Needs a restart of the script")
        style = "color: #9cdcfe;" if p.get("live") else ""
        rows.append(item(r, 0, label(f"lbl__{section}__{key}", p.get("label", key), tip, style)))
        if typ == "bool":
            w = widget("QCheckBox", wname, prop("toolTip", string(tip)))
        elif typ == "int":
            w = widget("QSpinBox", wname,
                       prop("minimum", f"<number>{p['min']}</number>")
                       + prop("maximum", f"<number>{p['max']}</number>")
                       + prop("singleStep", f"<number>{p.get('step', 1)}</number>")
                       + prop("keyboardTracking", "<bool>false</bool>")
                       + prop("toolTip", string(tip))
                       + (prop("enabled", "<bool>false</bool>") if p.get("readonly") else ""))
        else:
            w = widget("QDoubleSpinBox", wname,
                       prop("decimals", f"<number>{p.get('decimals', 3)}</number>")
                       + prop("minimum", f"<double>{p['min']}</double>")
                       + prop("maximum", f"<double>{p['max']}</double>")
                       + prop("singleStep", f"<double>{p.get('step', 0.01)}</double>")
                       + prop("keyboardTracking", "<bool>false</bool>")
                       + prop("toolTip", string(tip)))
        rows.append(item(r, 1, w))
        r += 1
    return group(f"grp__{section}", SECTIONS[section], grid(f"grid__{section}", rows))


def button(name, text, tip="", enabled=True):
    p = prop("text", string(text))
    if tip:
        p += prop("toolTip", string(tip))
    if not enabled:
        p += prop("enabled", "<bool>false</bool>")
    return widget("QPushButton", name, p)


# ---------------- left column: run, live control, prerequisites, log ----------------
run_items = [
    item(0, 0, label("scriptLabel", "Script:")),
    item(0, 1, label("scriptPathLabel", "---", style="color: #aaaaaa;"), 2),
    item(1, 0, label("runStateTitle", "State:")),
    item(1, 1, label("runStateLabel", "---", style="font-weight: bold;"), 2),
    item(2, 0, label("cpuTitle", "CPUs (taskset -c):")),
    item(2, 1, widget("QLineEdit", "cpuEdit",
                      prop("text", string("10-20"))
                      + prop("toolTip", string("CPU list for taskset, e.g. 10-20 or 4,5,10-12; empty = no pinning"))), 2),
    item(3, 0, label("condaTitle", "Conda env:")),
    item(3, 1, widget("QLineEdit", "condaEnvEdit",
                      prop("text", string("rtc"))
                      + prop("toolTip", string("Environment activated with 'conda activate' before "
                                               "running the script (torch + dao)"))), 2),
    item(4, 0, button("startButton", "START", "Launch po4ao_rama.py in the PO4AO_RL tmux session "
                                               "with the saved config (it drives the DM)")),
    item(4, 1, button("stopButton", "STOP", "Ctrl-C the PO4AO_RL session")),
    item(4, 2, button("logButton", "OPEN LOG", "Open a terminal attached to the tmux session")),
    item(5, 0, label("runCommandLabel", "", style="color: #888888; font-size: 11px;"), 3),
]
live_items = [
    item(0, 0, widget("QLabel", "liveStateLabel",
                      prop("text", string("Live control: not available (script not running)"))
                      + prop("wordWrap", "<bool>true</bool>")
                      + prop("styleSheet", string("color: #aaaaaa;"))), 3),
    item(1, 0, button("closeLoopButton", "CLOSE (policy)", "Policy drives dm1Cmd05 (from the next episode)", False)),
    item(1, 1, button("integratorButton", "INTEGRATOR", "Integrator drives dm1Cmd05 (from the next episode)", False)),
    item(1, 2, button("openLoopButton", "OPEN LOOP", "Flatten dm1Cmd05 now; the current episode is dropped", False)),
    item(2, 0, widget("QCheckBox", "trainingCheckBox",
                      prop("text", string("Training enabled"))
                      + prop("toolTip", string("Trainer process runs continuously on the replay buffers"))
                      + prop("enabled", "<bool>false</bool>")), 1),
    item(2, 1, widget("QCheckBox", "explorationCheckBox",
                      prop("text", string("Exploration noise"))
                      + prop("toolTip", string("Add max_sigma noise to the commands (from the next episode)"))
                      + prop("enabled", "<bool>false</bool>")), 1),
    item(2, 2, button("applyLiveButton", "APPLY LIVE", "Send the blue parameters to the running script", False)),
    item(3, 0, button("saveModelsButton", "SAVE MODELS", "Checkpoint dynamics + policy into the run directory", False)),
    item(3, 1, button("saveBuffersButton", "SAVE BUFFERS", "Checkpoint both replay buffers into the run directory", False)),
    item(3, 2, button("reloadPolicyButton", "RELOAD POLICY", "Swap the inference policy with the latest trained one", False)),
    item(4, 0, label("checkpointTitle", "Checkpoint dir:")),
    item(4, 1, widget("QLineEdit", "checkpointEdit",
                      prop("toolTip", string("Directory holding models.pt / replay.pt / replay_warmup.pt"))), 1),
    item(4, 2, button("browseCheckpointButton", "BROWSE...", "Pick a checkpoint directory")),
    item(5, 0, button("warmupButton", "WARM-UP", "Integrator + noise episodes, then pre-training, then the policy", False)),
    item(5, 1, button("loadCheckpointButton", "LOAD CHECKPOINT", "Load models / buffers from the directory above (loop opens)", False)),
]
prereq_items = [
    item(0, 0, label("preCnnTitle", "CNN modes (pyrModesNN):")),
    item(0, 1, label("preCnnLabel", "---")),
    item(1, 0, label("preNnLoopTitle", "NN integrator (lpCmdNN):")),
    item(1, 1, label("preNnLoopLabel", "---")),
    item(2, 0, label("preLinLoopTitle", "Linear integrator (lpCmd):")),
    item(2, 1, label("preLinLoopLabel", "---")),
    item(3, 0, label("preDmTitle", "RL channel (dm1Cmd05):")),
    item(3, 1, label("preDmLabel", "---")),
    item(4, 0, label("preGpuTitle", "GPU:")),
    item(4, 1, label("preGpuLabel", "---")),
]
status_items = [
    item(0, 0, label("budgetLabel", "---", style="color: #9cdcfe;")),
    item(1, 0, widget("QPlainTextEdit", "logText",
                      prop("readOnly", "<bool>true</bool>")
                      + prop("lineWrapMode", "<enum>QPlainTextEdit::NoWrap</enum>")
                      + prop("maximumBlockCount", "<number>2000</number>")
                      + prop("minimumSize", "<size><width>0</width><height>220</height></size>")
                      + prop("styleSheet", string("background-color: #151515; color: #cccccc; "
                                                   "font-family: monospace; font-size: 11px;")))),
]

left = ('<layout class="QVBoxLayout" name="leftColumn">'
        + "<item>" + group("runGroup", "PO4AO run", grid("runGrid", run_items)) + "</item>"
        + "<item>" + group("liveGroup", "Live control", grid("liveGrid", live_items)) + "</item>"
        + "<item>" + group("prereqGroup", "Bench prerequisites", grid("prereqGrid", prereq_items)) + "</item>"
        + '<item><spacer name="leftSpacer">' + prop("orientation", "<enum>Qt::Vertical</enum>") + '</spacer></item>'
        + "</layout>")

# ---------------- right column: parameters in two sub-columns + config buttons ----------------
col_a = ["RL", "integrator", "MDP"]
col_b = ["training", "replay_buffers", "NN_models", "save_and_load"]


def vcol(name, sections):
    # sections with no entry in PARAMS (removed from the GUI) are skipped
    sections = [s for s in sections if any(p["section"] == s for p in PARAMS)]
    return (f'<layout class="QVBoxLayout" name="{name}">'
            + "".join("<item>" + param_group(s) + "</item>" for s in sections)
            + '<item><spacer name="' + name + 'Spacer">'
            + prop("orientation", "<enum>Qt::Vertical</enum>")
            + '</spacer></item></layout>')


cfg_buttons = ('<layout class="QHBoxLayout" name="cfgButtons">'
               + "<item>" + label("configPathLabel", "---", style="color: #aaaaaa;") + "</item>"
               + '<item><spacer name="cfgSpacer">' + prop("orientation", "<enum>Qt::Horizontal</enum>") + "</spacer></item>"
               + "<item>" + button("reloadConfigButton", "RELOAD CONFIG", "Discard edits, re-read po4ao_config.py") + "</item>"
               + "<item>" + button("saveConfigButton", "SAVE CONFIG", "Write po4ao_config.py (backup kept, comments preserved)") + "</item>"
               + "</layout>")

legend = label("legendLabel",
               "blue = live (APPLY LIVE while the script runs); white = read at launch, needs a restart",
               style="color: #888888; font-size: 11px;")

right = ('<layout class="QVBoxLayout" name="rightColumn">'
         + '<item><layout class="QHBoxLayout" name="paramColumns">'
         + "<item>" + vcol("paramColA", col_a) + "</item>"
         + "<item>" + vcol("paramColB", col_b) + "</item>"
         + "</layout></item>"
         + "<item>" + legend + "</item>"
         + "<item>" + cfg_buttons + "</item>"
         + "</layout>")

central = widget("QWidget", "centralwidget", "",
                 '<layout class="QVBoxLayout" name="mainLayout">'
                 + '<item><layout class="QHBoxLayout" name="topLayout">'
                 + '<item>' + left + '</item>'
                 + '<item>' + right + '</item>'
                 + '</layout></item>'
                 + '<item>' + group("statusGroup", "Console (tmux PO4AO_RL)",
                                   grid("statusGrid", status_items)) + '</item>'
                 + '</layout>')

ui = ('<?xml version="1.0" encoding="UTF-8"?>\n<ui version="4.0">\n <class>MainWindow</class>\n'
      + widget("QMainWindow", "MainWindow",
               prop("geometry", "<rect><x>0</x><y>0</y><width>1520</width><height>980</height></rect>")
               + prop("windowTitle", string("RAMA PO4AO Control")),
               central)
      + "\n <resources/>\n <connections/>\n</ui>\n")
open(out, "w").write(ui)
print("written", out)
