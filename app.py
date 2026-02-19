import streamlit as st
import subprocess
import json
import os
import threading
import time
import sys

# Page Config
st.set_page_config(
    page_title="Visual Desktop Control",
    page_icon="🖐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title & Styling
st.title("🖐 Visual Desktop Control Center")
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Path to python executable in venv
if sys.platform == "win32":
    PYTHON_EXE = os.path.join("venv", "Scripts", "python.exe")
else:
    PYTHON_EXE = os.path.join("venv", "bin", "python")

CONFIG_FILE = "gestures_config.json"

# --- Functions ---

def load_config():
    if not os.path.exists(CONFIG_FILE):
        return {"gestures": []}
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)

def save_config(data):
    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=4)

def run_process(script_name):
    """Runs a script in a subprocess and streams output."""
    cmd = [PYTHON_EXE, script_name]
    
    # Check if file exists
    if not os.path.exists(script_name):
        st.error(f"File not found: {script_name}")
        return

    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
    )

# --- Sidebar: Configuration ---
st.sidebar.header("⚙️ Configuration")

config = load_config()
gestures = config.get("gestures", [])

st.sidebar.subheader("Gesture Mapping")
st.sidebar.info("Modify 'gestures_config.json' to add new classes.")

# Display current gestures
for i, g in enumerate(gestures):
    st.sidebar.text(f"{i}: {g}")

# Add new gesture (Simple append for now)
new_gesture = st.sidebar.text_input("New Gesture Name")
if st.sidebar.button("Add New Gesture"):
    if new_gesture and new_gesture not in gestures:
        gestures.append(new_gesture)
        config["gestures"] = gestures
        save_config(config)
        st.sidebar.success(f"Added {new_gesture}! Scripts updated.")
        time.sleep(1)
        st.rerun()
    elif new_gesture in gestures:
        st.sidebar.warning("Gesture already exists.")

if st.sidebar.button("Reset Config (Default)"):
    config["gestures"] = ["Volume", "Bright_Up", "Bright_Down", "Show_Desktop"]
    save_config(config)
    st.sidebar.success("Reset to defaults.")
    time.sleep(1)
    st.rerun()

# --- Main Dashboard ---

col1, col2, col3 = st.columns(3)

with col1:
    st.header("1. Capture Data")
    st.write("Record hand landmarks for your defined gestures.")
    if st.button("📷 Start Data Collection", type="primary"):
        st.info("Launching 'collect_data.py'... Check the new window.")
        run_process("collect_data.py")

with col2:
    st.header("2. Train Model")
    st.write("Train the Neural Network on collected data.")
    if st.button("🧠 Start Training"):
        with st.status("Training Model...", expanded=True) as status:
            process = subprocess.Popen(
                [PYTHON_EXE, "train_model.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            output_container = st.empty()
            full_logs = ""
            
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    full_logs += line
                    output_container.code(line.strip(), language="text")
            
            if process.returncode == 0:
                status.update(label="Training Complete!", state="complete", expanded=False)
                st.success("Model Saved: gesture_model.h5")
            else:
                status.update(label="Training Failed", state="error")
                st.error("Check logs.")

with col3:
    st.header("3. Run Control")
    st.write("Start the real-time control system.")
    if st.button("🚀 Start System", type="primary"):
        st.info("System Running. Press 'q' in the camera window to stop.")
        run_process("main.py")

# --- Logs Section ---
st.divider()
st.subheader("📋 System Actions")
with st.expander("Show Instructions"):
    st.markdown("""
    1.  **Add Gestures** in the sidebar if needed.
    2.  **Collect Data**: Click 'Start Data Collection'. A window will open. Press keys **0-N** to record. Press **q** to quit.
    3.  **Train Model**: Click 'Start Training'. Wait for it to finish.
    4.  **Run System**: Click 'Start System'. Control your PC!
    """)
