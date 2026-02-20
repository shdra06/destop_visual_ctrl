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

# Custom CSS for Modern UI
st.markdown("""
<style>
    /* Reduce wasted space at the top */
    .block-container {
        padding-top: 1.5rem !important;
        max-width: 1200px !important;
    }

    /* Gradient text for main title */
    .main-title {
        background: -webkit-linear-gradient(45deg, #FF4B2B, #FF416C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        margin-bottom: 0rem;
        padding-bottom: 0rem;
        line-height: 1.2;
    }
    .sub-title {
        color: #a0a0a0;
        font-size: 1.15rem;
        margin-top: 0.5rem;
        margin-bottom: 2.5rem;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    
    /* Make the pipeline cards "pop" and feel alive */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.3s ease, border-color 0.3s ease !important;
        border-radius: 12px !important;
        background: rgba(30, 30, 30, 0.6) !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4) !important; /* Elevated base state */
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-6px);
        box-shadow: 0 16px 24px rgba(255, 75, 43, 0.15), 0 0 10px rgba(255, 75, 43, 0.05) !important;
        border-color: #FF4B2B !important;
    }

    /* Better buttons */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.2em;
        font-weight: 600;
        transition: all 0.3s ease !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background-color: rgba(40, 40, 40, 0.8);
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.3) !important;
        border-color: rgba(255, 255, 255, 0.3);
        color: white;
    }
    .stButton>button:active {
        transform: translateY(0px) scale(0.98);
    }
    
    /* Primary buttons get a modern glow */
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #FF4B2B 0%, #FF416C 100%);
        border: none;
        color: white;
    }
    .stButton>button[kind="primary"]:hover {
        box-shadow: 0 8px 20px rgba(255, 65, 108, 0.4) !important;
        filter: brightness(1.1);
    }

    /* Status indicators */
    .status-online { color: #00E676; font-weight: 800; padding: 6px 12px; background: rgba(0,230,118,0.15); border-radius: 6px; letter-spacing: 1px; display: inline-block; margin-top: 5px; }
    .status-offline { color: #FF1744; font-weight: 800; padding: 6px 12px; background: rgba(255,23,68,0.15); border-radius: 6px; letter-spacing: 1px; display: inline-block; margin-top: 5px; }
    
    /* Sidebar styling & refinement */
    section[data-testid="stSidebar"] {
        background-color: rgba(18, 18, 18, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    .sidebar-header {
        font-size: 1.3rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        color: #f0f0f0;
        border-bottom: 2px solid rgba(255, 75, 43, 0.5);
        padding-bottom: 5px;
        display: inline-block;
    }
    
    hr {
        border-color: rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🖐 Visual Desktop Control</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Train and run hand gesture recognition to control your computer seamlessly.</p>', unsafe_allow_html=True)

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

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.markdown("## ⚙️ Control Center")
    st.markdown("---")
    
    config = load_config()
    gestures = config.get("gestures", [])
    
    with st.expander("🎯 Gesture Mapping", expanded=True):
        st.caption("Current active gestures:")
        for i, g in enumerate(gestures):
            st.markdown(f"**{i}**: `{g}`")
            
        st.divider()
        new_gesture = st.text_input("Register New Gesture", placeholder="e.g., Swipe_Left")
        if st.button("➕ Add Gesture", use_container_width=True):
            if new_gesture and new_gesture not in gestures:
                gestures.append(new_gesture)
                config["gestures"] = gestures
                save_config(config)
                st.success(f"Added {new_gesture}!")
                time.sleep(1)
                st.rerun()
            elif new_gesture in gestures:
                st.warning("Gesture already exists.")

        if st.button("🔄 Reset Defaults", use_container_width=True):
            config["gestures"] = ["Volume", "Bright_Up", "Bright_Down", "Show_Desktop"]
            save_config(config)
            st.success("Reset to defaults.")
            time.sleep(1)
            st.rerun()

    with st.expander("🖱️ Mouse & Input Mode", expanded=False):
        mouse_mode = config.get("mouse_mode", False)
        new_mouse_mode = st.toggle("Enable Air Mouse Mode", value=mouse_mode, help="2-Hand Control: Left=Cursor, Right=Click")
        
        if new_mouse_mode != mouse_mode:
            config["mouse_mode"] = new_mouse_mode
            save_config(config)
            st.toast(f"Mouse Mode {'Enabled' if new_mouse_mode else 'Disabled'}!")
            time.sleep(0.5)
            st.rerun()

    with st.expander("🖥️ Window Settings", expanded=False):
        headless = config.get("headless", False)
        always_on_top = config.get("always_on_top", True)
        
        new_headless = st.toggle("Headless Mode", value=headless, help="Hides camera view, runs in background.")
        new_always_on_top = st.toggle("Always On Top", value=always_on_top, help="Keeps camera window above others.")
        
        if new_headless != headless or new_always_on_top != always_on_top:
            config["headless"] = new_headless
            config["always_on_top"] = new_always_on_top
            save_config(config)
            st.toast("Window settings updated!")
            time.sleep(0.5)
            st.rerun()

    st.markdown("---")
    if st.button("🔌 Restart Dashboard", type="primary", use_container_width=True):
        st.warning("Restarting dashboard...")
        time.sleep(1)
        os.execv(sys.executable, [sys.executable, "-m", "streamlit", "run", "app.py"])


# --- MAIN PIPELINE LAYOUT (2x2 Grid) ---
st.markdown("### 🚀 Engine Pipeline")
col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.subheader("📷 1. Data Collection")
        st.markdown("Record custom hand gestures to build your dataset.")
        if st.button("Start Data Collector", key="btn_collect"):
            st.toast("Launched data collector in background terminal.")
            run_process("collect_data.py", capture_output=False)

with col2:
    with st.container(border=True):
        st.subheader("🧠 2. Model Training")
        st.markdown("Train the neural network on your collected data.")
        if st.button("Train AI Model", key="btn_train"):
            with st.status("Training in progress...", expanded=True) as status:
                process = subprocess.Popen(
                    [PYTHON_EXE, "train_model.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                output_container = st.empty()
                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    if line:
                        output_container.code(line.strip(), language="text")
                if process.returncode == 0:
                    status.update(label="Training Complete! ✅", state="complete", expanded=False)
                    st.success("Model saved successfully as: `gesture_model.h5`")
                else:
                    status.update(label="Training Failed ❌", state="error")
                    st.error("Check logs for details.")

st.markdown("<br>", unsafe_allow_html=True)
col3, col4 = st.columns(2)

with col3:
    with st.container(border=True):
        st.subheader("🧪 3. System Testing")
        st.markdown("Debug and verify gesture recognition accuracy.")
        if st.button("Run Test Viewer", key="btn_test"):
             st.toast("Launched test viewer in background terminal.")
             run_process("test_gesture.py", capture_output=False)

if 'system_pid' not in st.session_state:
    st.session_state['system_pid'] = None

with col4:
    with st.container(border=True):
        st.subheader("⚡ 4. Live Control")
        
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
            
        if is_running:
            st.markdown("Status: <span class='status-online'>🟢 ONLINE</span>", unsafe_allow_html=True)
            st.markdown(f"*Engine running in background (PID: {pid})*")
            
            c4_1, c4_2 = st.columns(2)
            with c4_1:
                if st.button("⏹ Stop System"):
                    try:
                        p = psutil.Process(pid)
                        p.kill() 
                        p.wait(timeout=3) 
                        st.session_state['system_pid'] = None
                        st.rerun()
                    except (psutil.NoSuchProcess, Exception) as e:
                        st.session_state['system_pid'] = None
                        st.rerun()
            with c4_2:
                if st.button("🔄 Restart"):
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
        else:
            st.markdown("Status: <span class='status-offline'>🔴 OFFLINE</span>", unsafe_allow_html=True)
            st.markdown("*Start the engine to control your desktop.*")
            if st.button("🚀 Start Engine", key="btn_start", type="primary"):
                st.toast("Initializing Visual Desktop Control...")
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

st.divider()

with st.expander("📖 Getting Started & Best Practices", expanded=True):
    st.markdown("### 🎯 Interactive Gesture Library")
    st.markdown("Edit, add, or delete gesture details below. Click **Save** when done.")
    
    # Default library if none in config
    default_lib = [
        {"Gesture": "Volume", "Shape": "👌 Pinch", "Action": "Adjusts Volume", "Motion": "Move UP/DOWN"},
        {"Gesture": "Bright_Up", "Shape": "👍 Thumb Up", "Action": "Increases Brightness", "Motion": "Hold"},
        {"Gesture": "Bright_Down", "Shape": "👎 Thumb Down", "Action": "Decreases Brightness", "Motion": "Hold"},
        {"Gesture": "Show_Desktop", "Shape": "🤘 Rock", "Action": "Minimizes windows", "Motion": "Hold"},
        {"Gesture": "Idle", "Shape": "🖐️ Relaxed", "Action": "None", "Motion": "None"},
        {"Gesture": "Victory", "Shape": "✌️ Peace", "Action": "Toggle Mouse/Gesture", "Motion": "Hold"},
        {"Gesture": "pause track", "Shape": "🤏 Open Pinch", "Action": "Play/Pause Media", "Motion": "Hold 0.5s"},
        {"Gesture": "Scroll Mode", "Shape": "☝️ Thumb Extended", "Action": "Scroll Page (Mouse Mode)", "Motion": "Move Index"}
    ]
    
    library_data = config.get("gesture_library", default_lib)
    
    edited_data = st.data_editor(
        library_data, 
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True
    )
    
    if st.button("💾 Save Library Changes", key="save_lib"):
        config["gesture_library"] = edited_data
        save_config(config)
        st.success("Library updated successfully!")

    st.markdown("""
    ### 🛠️ Step-by-Step Guide
    1. **Data Collection**: Stand in a well-lit area. Click `Start Data Collector` and follow the prompt. Capture at least ~1000 frames per gesture at various angles.
    2. **Training**: Once data is collected, hit `Train AI Model`. Wait for the success message.
    3. **Testing**: Use `Run Test Viewer` to ensure your gestures are recognized efficiently without drift.
    4. **Execution**: Click `Start Engine` to begin controlling your PC dynamically!
    """)
