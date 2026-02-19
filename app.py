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

def run_process(script_name, capture_output=False):
    """Runs a script in a subprocess and streams output."""
    cmd = [PYTHON_EXE, script_name]
    
    # Check if file exists
    if not os.path.exists(script_name):
        st.error(f"File not found: {script_name}")
        return

    kwargs = {
        'creationflags': subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0,
        'text': True
    }

    if capture_output:
        kwargs['stdout'] = subprocess.PIPE
        kwargs['stderr'] = subprocess.STDOUT
        kwargs['bufsize'] = 1

    return subprocess.Popen(cmd, **kwargs)

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

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Restart Dashboard", type="primary"):
    st.sidebar.warning("Restarting...")
    time.sleep(1)
    # Reload the streamlit process
    os.execv(sys.executable, [sys.executable, "-m", "streamlit", "run", "app.py"])

# --- Main Dashboard ---

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.header("1. Capture")
    st.write("Record data.")
    if st.button("📷 Collect Data", key="btn_collect", type="primary"):
        st.info("Running collect_data.py in new terminal...")
        run_process("collect_data.py", capture_output=False)

with col2:
    st.header("2. Train")
    st.write("Train model.")
    if st.button("🧠 Train Model", key="btn_train"):
        with st.status("Training...", expanded=True) as status:
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
                status.update(label="Done!", state="complete", expanded=False)
                st.success("Saved: gesture_model.h5")
            else:
                status.update(label="Failed", state="error")
                st.error("Check logs.")

with col3:
    st.header("3. Test")
    st.write("Debug detection.")
    if st.button("🧪 Test Model", key="btn_test"):
        st.info("Running test_gesture.py in new terminal...")
        run_process("test_gesture.py", capture_output=False)

# --- Session State for Process Management ---
if 'system_pid' not in st.session_state:
    st.session_state['system_pid'] = None

with col4:
    st.header("4. Run")
    st.write("Control System.")
    
    # Check if running
    is_running = st.session_state['system_pid'] is not None
    
    if not is_running:
        if st.button("🚀 Start System", key="btn_start", type="primary"):
            st.info("Starting main.py...")
            proc = run_process("main.py", capture_output=False)
            if proc:
                st.session_state['system_pid'] = proc.pid
                st.rerun()
    else:
        st.success(f"System Running (PID: {st.session_state['system_pid']})")
        
        if st.button("⏹ Stop System"):
            try:
                os.kill(st.session_state['system_pid'], 9) # Force kill
                st.session_state['system_pid'] = None
                st.rerun()
            except Exception as e:
                st.error(f"Error stopping: {e}")
                st.session_state['system_pid'] = None # Reset anyway
        
        if st.button("🔄 Restart System"):
             try:
                os.kill(st.session_state['system_pid'], 9)
                time.sleep(1)
                proc = run_process("main.py", capture_output=False)
                if proc:
                    st.session_state['system_pid'] = proc.pid
                st.rerun()
             except Exception as e:
                st.error(f"Error restarting: {e}")

# --- Logs Section ---
st.divider()
st.subheader("📋 Guide & Instructions")
with st.expander("Show Detailed Guide", expanded=True):
    st.markdown("""
    ### 🖐 Recommended Hand Shapes (Updated)
    1.  **Volume (0)**: **Pinch (👌)**. Index + Thumb closed. Move Hand UP/DOWN.
    2.  **Bright_Up (1)**: **Thumb Up (👍)**. Hand closed with Thumb UP.
    3.  **Bright_Down (2)**: **Thumb Down (👎)**. Hand closed with Thumb DOWN.
    4.  **Show_Desktop (3)**: **Rock (🤘)** or **Thumb + 2 Fingers**.
    5.  **Idle (4)**: **Relaxed Hand / Random Motion**.

    ### 📝 Workflow
    1.  **Collect Data**: Click 'Collect Data'. Record ~1000 frames for EACH gesture above.
    2.  **Train Model**: Click 'Train Model'.
    3.  **Test**: Use 'Test Model' to verify recognition.
    4.  **Run**: Click 'Start System'.
    """)
