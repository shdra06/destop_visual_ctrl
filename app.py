import streamlit as st
import subprocess
import json
import os
import threading
import time
import sys
import psutil


st.set_page_config(
    page_title="Visual Desktop Control",
    page_icon="🖐",
    layout="wide",
    initial_sidebar_state="expanded"
)


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


if sys.platform == "win32":
    PYTHON_EXE = os.path.join("venv", "Scripts", "python.exe")
else:
    PYTHON_EXE = os.path.join("venv", "bin", "python")

CONFIG_FILE = "gestures_config.json"


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


st.sidebar.header("⚙️ Configuration")

config = load_config()
gestures = config.get("gestures", [])

st.sidebar.subheader("Gesture Mapping")
st.sidebar.info("Modify 'gestures_config.json' to add new classes.")


for i, g in enumerate(gestures):
    st.sidebar.text(f"{i}: {g}")


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

mouse_mode = config.get("mouse_mode", False)
new_mouse_mode = st.sidebar.checkbox("🖱 Enable Air Mouse Mode", value=mouse_mode, help="2-Hand Control: Left=Cursor, Right=Click")

if new_mouse_mode != mouse_mode:
    config["mouse_mode"] = new_mouse_mode
    save_config(config)
    st.sidebar.info(f"Mouse Mode {'Enabled' if new_mouse_mode else 'Disabled'}")
    time.sleep(0.5)
    st.rerun()


st.sidebar.markdown("### 🖥️ Window Settings")
headless = config.get("headless", False)
always_on_top = config.get("always_on_top", True)

new_headless = st.sidebar.checkbox("🚫 Headless Mode (No Window)", value=headless, help="Fixes background lag but hides camera view.")
new_always_on_top = st.sidebar.checkbox("🔝 Always On Top", value=always_on_top, help="Keeps window visible to prevent throttling.")

if new_headless != headless or new_always_on_top != always_on_top:
    config["headless"] = new_headless
    config["always_on_top"] = new_always_on_top
    save_config(config)
    st.rerun()

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Restart Dashboard", type="primary"):
    st.sidebar.warning("Restarting...")
    time.sleep(1)
   
    os.execv(sys.executable, [sys.executable, "-m", "streamlit", "run", "app.py"])



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


if 'system_pid' not in st.session_state:
    st.session_state['system_pid'] = None

with col4:
    st.header("4. Run")
    st.write("Control System.")
    
    
    pid = st.session_state.get('system_pid', None)
    is_running = False
    
    if pid:
        if psutil.pid_exists(pid):
            try:
                p = psutil.Process(pid)
                
                if p.status() != psutil.STATUS_ZOMBIE:
                    is_running = True
            except psutil.NoSuchProcess:
                is_running = False
        else:
            is_running = False
            
   
    if not is_running and pid is not None:
        st.session_state['system_pid'] = None
        st.warning(f"Process {pid} ended unexpectedly.")

    if not is_running:
        if st.button("🚀 Start System", key="btn_start", type="primary"):
            st.info("Starting main.py...")
            try:
                proc = run_process("main.py", capture_output=False)
                if proc:
                    st.session_state['system_pid'] = proc.pid
                    time.sleep(1) 
                    st.rerun()
                else:
                    st.error("Failed to start process.")
            except Exception as e:
                st.error(f"Start Error: {e}")
                
    else:
        st.success(f"System Running (PID: {pid})")
        
        if st.button("⏹ Stop System"):
            try:
                p = psutil.Process(pid)
                p.kill() 
                p.wait(timeout=3) 
                st.session_state['system_pid'] = None
                st.rerun()
            except psutil.NoSuchProcess:
                st.session_state['system_pid'] = None
                st.rerun()
            except Exception as e:
                st.error(f"Error stopping: {e}")
        
        if st.button("🔄 Restart System"):
             try:
                try:
                    p = psutil.Process(pid)
                    p.kill()
                    p.wait(timeout=3)
                except psutil.NoSuchProcess:
                    pass
                
                time.sleep(1)
                proc = run_process("main.py", capture_output=False)
                if proc:
                    st.session_state['system_pid'] = proc.pid
                st.rerun()
             except Exception as e:
                st.error(f"Error restarting: {e}")


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
