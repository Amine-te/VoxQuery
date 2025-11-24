"""
VoxQuery - Voice to SQL Streamlit Application
A modern interface for voice-powered database querying
"""

import streamlit as st
import time
import queue
import tempfile
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from typing import Optional, Dict
import json

# Import your RAG engine
from rag_engine import RAGEngine

# ============================================================================
# CONFIGURATION & STYLING
# ============================================================================

def apply_dark_theme():
    """Apply VS Code-inspired dark theme"""
    st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #1e1e1e;
        color: #d4d4d4;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #252526;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #4ec9b0 !important;
        font-weight: 600;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #4fc3f7;
        font-size: 28px;
    }
    
    [data-testid="stMetricLabel"] {
        color: #9cdcfe;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background-color: #1e1e1e;
        border: 1px solid #3c3c3c;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: #1e3a1e;
        color: #4ec9b0;
    }
    
    .stError {
        background-color: #3a1e1e;
        color: #f48771;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #0e639c;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: background-color 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #1177bb;
    }
    
    /* Text input */
    .stTextInput>div>div>input {
        background-color: #3c3c3c;
        color: #d4d4d4;
        border: 1px solid #6a6a6a;
    }
    
    /* Text area */
    .stTextArea>div>div>textarea {
        background-color: #3c3c3c;
        color: #d4d4d4;
        border: 1px solid #6a6a6a;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #2d2d30;
        color: #cccccc;
    }
    
    /* Divider */
    hr {
        border-color: #3c3c3c;
    }
    
    /* Recording indicator */
    .recording-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        background-color: #f44336;
        border-radius: 50%;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
    
    /* Custom containers */
    .metric-container {
        background-color: #2d2d30;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #3c3c3c;
        margin: 0.5rem 0;
    }
    
    .debug-section {
        background-color: #252526;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007acc;
        margin: 1rem 0;
    }
    
    /* Table styling */
    .dataframe {
        background-color: #1e1e1e !important;
    }
    
    .dataframe th {
        background-color: #2d2d30 !important;
        color: #9cdcfe !important;
    }
    
    .dataframe td {
        background-color: #1e1e1e !important;
        color: #d4d4d4 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #2d2d30;
        color: #cccccc;
        border-radius: 4px 4px 0 0;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0e639c;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# AUDIO RECORDING CLASS
# ============================================================================

class AudioRecorder:
    """Real-time audio recorder for voice input"""
    
    def __init__(self, samplerate: int = 16000):
        self.samplerate = samplerate
        self.recording = False
        self.audio_queue = queue.Queue()
        self.stream = None
    
    def callback(self, indata, frames, time_info, status):
        """Audio callback for continuous capture"""
        if self.recording:
            self.audio_queue.put(indata.copy())
    
    def start(self):
        """Start recording"""
        self.recording = True
        self.audio_queue = queue.Queue()
        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=1,
            callback=self.callback,
            dtype='float32'
        )
        self.stream.start()
    
    def stop(self):
        """Stop recording and return audio data"""
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        
        chunks = []
        while not self.audio_queue.empty():
            chunks.append(self.audio_queue.get())
        
        return np.concatenate(chunks, axis=0) if chunks else None


# ============================================================================
# MODEL LOADING & CACHING
# ============================================================================

@st.cache_resource
def load_whisper_model(model_name: str = "base"):
    """Load and cache Whisper model"""
    import whisper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name, device=device)
    return model, device


def create_rag_engine(db_config: Dict, ollama_model: str, n_tables: int, n_examples: int):
    """Create a new RAG engine instance (not cached)"""
    return RAGEngine(
        db_config=db_config,
        ollama_model=ollama_model,
        n_tables=n_tables,
        n_data_per_table=n_examples,
        verbose=True
    )


# ============================================================================
# AUDIO PROCESSING
# ============================================================================

def transcribe_audio(audio_path: str, whisper_model, language: Optional[str] = None) -> tuple:
    """Transcribe audio file using Whisper"""
    start = time.time()
    if language and language != "Auto-detect":
        result = whisper_model.transcribe(audio_path, language=language.lower())
    else:
        result = whisper_model.transcribe(audio_path)
    elapsed = time.time() - start
    return result["text"].strip(), elapsed


def save_audio(audio_data: np.ndarray, samplerate: int) -> str:
    """Save audio data to temporary WAV file"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        sf.write(tmp.name, audio_data, samplerate)
        return tmp.name


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'input_mode': 'voice',  # 'voice' or 'text'
        'recorder': None,
        'is_recording': False,
        'recording_start_time': None,
        'recorded_audio': None,
        'audio_path': None,
        'transcription': None,
        'transcription_time': None,
        'sql_query': None,
        'query_results': None,
        'debug_info': None,
        'rag_engine': None,
        'whisper_model': None,
        'whisper_device': None,
        'total_pipeline_time': None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_sidebar():
    """Render sidebar with configuration options"""
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # ASR Settings
        st.subheader("🎤 ASR Settings")
        asr_model = st.selectbox(
            "Whisper Model",
            ["tiny", "base", "small", "medium", "large"],
            index=1,
            help="Larger models are more accurate but slower"
        )
        
        # Language selection
        language = st.selectbox(
            "Language",
            ["Auto-detect", "English", "French", "Spanish", "Arabic", "German", 
             "Italian", "Portuguese", "Russian", "Japanese", "Chinese"],
            index=0,
            help="Force ASR to use a specific language"
        )
        
        st.divider()
        
        # LLM Settings
        st.subheader("🤖 LLM Settings")
        llm_model = st.selectbox(
            "Ollama Model",
            ["llama3:8b", "mistral:latest", "codellama:7b"],
            index=0,
            help="Select your Ollama model"
        )
        
        st.divider()
        
        # RAG Settings
        st.subheader("🔍 RAG Settings")
        n_tables = st.slider(
            "Number of Tables to Retrieve",
            min_value=1,
            max_value=10,
            value=3,
            help="How many relevant tables to retrieve in Stage 1"
        )
        
        n_examples = st.slider(
            "Examples per Table",
            min_value=1,
            max_value=20,
            value=1,
            help="Number of data examples to retrieve per table"
        )
        
        show_prompt = st.checkbox(
            "Show Full Prompt in Debug",
            value=False,
            help="Display the complete LLM prompt in debug section"
        )
        
        st.divider()
        
        # Database Settings
        st.subheader("💾 Database Settings")
        db_host = st.text_input("Host", value="localhost")
        db_user = st.text_input("User", value="root")
        db_password = st.text_input("Password", value="", type="password")
        db_name = st.text_input("Database", value="oncf_db")
        
        st.divider()
        
        # GPU Info
        st.subheader("🖥️ System Info")
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            st.success(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        else:
            st.warning("⚠️ Running on CPU")
        
        return {
            'asr_model': asr_model,
            'language': language,
            'llm_model': llm_model,
            'n_tables': n_tables,
            'n_examples': n_examples,
            'show_prompt': show_prompt,
            'db_config': {
                'host': db_host,
                'user': db_user,
                'password': db_password,
                'database': db_name
            }
        }


def render_voice_input():
    """Render voice input tab"""
    st.markdown("### 🎤 Voice Input Mode")
    st.markdown("Record your question or upload an audio file")
    
    st.markdown("---")
    
    # Two main sections side by side
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("#### 🔴 Record Audio")
        if st.button(
            "🔴 Start Recording" if not st.session_state.is_recording else "⏹️ Stop Recording",
            use_container_width=True,
            key="record_button"
        ):
            if not st.session_state.is_recording:
                # Start recording
                st.session_state.recorder = AudioRecorder()
                st.session_state.recorder.start()
                st.session_state.is_recording = True
                st.session_state.recording_start_time = time.time()
                st.rerun()
            else:
                # Stop recording
                audio_data = st.session_state.recorder.stop()
                st.session_state.is_recording = False
                st.session_state.recorded_audio = audio_data
                
                if audio_data is not None:
                    audio_path = save_audio(audio_data, st.session_state.recorder.samplerate)
                    st.session_state.audio_path = audio_path
                    st.success("✅ Recording saved!")
                
                st.rerun()
        
        # Recording indicator
        if st.session_state.is_recording:
            elapsed = time.time() - st.session_state.recording_start_time
            st.markdown(f"""
            <div style='text-align: center; padding: 1rem; background-color: #2d2d30; border-radius: 8px; border: 2px solid #f44336; margin-top: 1rem;'>
                <span class='recording-indicator'></span>
                <span style='color: #f44336; margin-left: 10px; font-weight: 600;'>
                    RECORDING - {elapsed:.1f}s
                </span>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.1)
            st.rerun()
    
    with col_right:
        st.markdown("#### 📁 Upload Audio File")
        uploaded_file = st.file_uploader(
            "Upload audio file",
            type=['wav', 'mp3', 'flac', 'm4a', 'ogg'],
            label_visibility="collapsed",
            key="audio_uploader"
        )
        
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                tmp.write(uploaded_file.read())
                st.session_state.audio_path = tmp.name
            st.success("✅ File uploaded!")
    
    # Audio playback and clear button
    if st.session_state.audio_path and not st.session_state.is_recording:
        st.markdown("---")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("#### 🔊 Audio Preview")
            st.audio(st.session_state.audio_path, format='audio/wav')
        
        with col2:
            st.markdown("####  ")  # Spacing to align button
            if st.button("🗑️ Clear Audio", use_container_width=True, key="clear_audio"):
                st.session_state.audio_path = None
                st.session_state.recorded_audio = None
                st.session_state.transcription = None
                st.session_state.transcription_time = None
                st.session_state.sql_query = None
                st.session_state.query_results = None
                st.session_state.debug_info = None
                st.rerun()


def render_text_input():
    """Render text input tab"""
    st.markdown("### ⌨️ Text Input Mode")
    st.markdown("Type your database question directly")
    
    st.markdown("---")
    
    text_query = st.text_area(
        "Enter your question",
        placeholder="e.g., Show me all trains departing from Casablanca today",
        height=200,
        key="text_query_input"
    )
    
    return text_query


def render_results():
    """Render results section"""
    if st.session_state.transcription:
        st.subheader("📝 Transcription")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(st.session_state.transcription)
        with col2:
            st.metric("ASR Time", f"{st.session_state.transcription_time:.2f}s")
    
    if st.session_state.sql_query:
        st.subheader("🔍 Generated SQL")
        st.code(st.session_state.sql_query, language="sql")
    
    if st.session_state.query_results:
        results = st.session_state.query_results
        
        st.subheader("📊 Query Results")
        
        if results['success']:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", results['row_count'])
            with col2:
                st.metric("Execution Time", f"{results['execution_time']:.3f}s")
            with col3:
                st.metric("Status", "✅ Success")
            
            if results['results']:
                st.dataframe(results['results'], use_container_width=True)
        else:
            st.error(f"❌ Execution failed: {results.get('error', 'Unknown error')}")


def render_debug_info(show_full_prompt: bool):
    """Render debug information"""
    if not st.session_state.debug_info:
        return
    
    debug = st.session_state.debug_info
    
    with st.expander("🔍 Debug Information", expanded=False):
        
        # Timing breakdown
        st.markdown("### ⏱️ Timing Breakdown")
        
        # Calculate total time including transcription if available
        total_time = (
            debug['stage1_time'] + 
            debug['stage2_time'] + 
            debug['generation_time'] + 
            debug['execution_time']
        )
        
        if st.session_state.transcription_time:
            total_time += st.session_state.transcription_time
            cols = st.columns(5)
            with cols[0]:
                st.metric("Transcription", f"{st.session_state.transcription_time:.2f}s")
            with cols[1]:
                st.metric("Stage 1", f"{debug['stage1_time']:.3f}s")
            with cols[2]:
                st.metric("Stage 2", f"{debug['stage2_time']:.3f}s")
            with cols[3]:
                st.metric("Generation", f"{debug['generation_time']:.2f}s")
            with cols[4]:
                st.metric("Execution", f"{debug['execution_time']:.3f}s")
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Stage 1", f"{debug['stage1_time']:.3f}s")
            with col2:
                st.metric("Stage 2", f"{debug['stage2_time']:.3f}s")
            with col3:
                st.metric("Generation", f"{debug['generation_time']:.2f}s")
            with col4:
                st.metric("Execution", f"{debug['execution_time']:.3f}s")
        
        # Total time
        st.markdown(f"**Total Pipeline Time: {total_time:.2f}s**")
        
        st.divider()
        
        # Retrieved tables
        st.markdown("### 📊 Stage 1: Retrieved Tables")
        for i, table in enumerate(debug['stage1_tables'], 1):
            with st.container():
                st.markdown(f"**{i}. {table['table_name']}**")
                st.caption(table['summary'])
                st.caption(f"Columns: {len(table['schema']['columns'])} | "
                          f"Foreign Keys: {len(table['schema']['foreign_keys'])}")
        
        st.divider()
        
        # Retrieved examples
        st.markdown("### 📦 Stage 2: Retrieved Data Examples")
        st.caption(f"Total examples: {debug['retrieved_example_count']}")
        for i, example in enumerate(debug['stage2_data'][:10], 1):
            st.text(f"{i}. {example}")
        
        if len(debug['stage2_data']) > 10:
            st.caption(f"... and {len(debug['stage2_data']) - 10} more")
        
        st.divider()
        
        # Prompt statistics
        st.markdown("### 📝 Prompt Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prompt Length", f"{len(debug['final_prompt'])} chars")
            st.metric("Tables in Context", debug['retrieved_table_count'])
        with col2:
            st.metric("Est. Tokens", f"~{len(debug['final_prompt']) // 4}")
            st.metric("Examples in Context", debug['retrieved_example_count'])
        
        # Full prompt if enabled
        if show_full_prompt:
            st.divider()
            st.markdown("### 📄 Full Prompt")
            st.code(debug['final_prompt'], language="text")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.set_page_config(
        page_title="VoxQuery - Voice to SQL",
        page_icon="🎤",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    apply_dark_theme()
    init_session_state()
    
    # Header
    st.title("🎤 VoxQuery")
    st.markdown("### Voice-Powered Database Querying with RAG")
    st.markdown("---")
    
    # Sidebar configuration
    config = render_sidebar()
    
    # Input mode tabs
    tab1, tab2 = st.tabs(["🎤 Voice Input", "⌨️ Text Input"])
    
    input_text = None
    
    with tab1:
        render_voice_input()
        st.session_state.input_mode = 'voice'
    
    with tab2:
        text_query = render_text_input()
        if text_query:
            input_text = text_query
            st.session_state.input_mode = 'text'
    
    st.markdown("---")
    
    # Process button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        process_button = st.button("🚀 Process Query", use_container_width=True, type="primary")
    
    if process_button:
        # Clear previous results
        st.session_state.sql_query = None
        st.session_state.query_results = None
        st.session_state.debug_info = None
        st.session_state.total_pipeline_time = None
        
        # Determine input source
        if st.session_state.input_mode == 'text' and input_text:
            st.info("📝 Using text input")
            # Clear transcription when using text input
            st.session_state.transcription = None
            st.session_state.transcription_time = None
        elif st.session_state.input_mode == 'voice' and st.session_state.audio_path:
            st.info("🎤 Transcribing audio...")
            
            # Load Whisper if needed
            if st.session_state.whisper_model is None or \
               st.session_state.get('current_asr_model') != config['asr_model']:
                with st.spinner(f"Loading Whisper {config['asr_model']} model..."):
                    st.session_state.whisper_model, st.session_state.whisper_device = \
                        load_whisper_model(config['asr_model'])
                    st.session_state.current_asr_model = config['asr_model']
            
            # Transcribe
            with st.spinner("Transcribing..."):
                transcription, transcription_time = transcribe_audio(
                    st.session_state.audio_path,
                    st.session_state.whisper_model,
                    config['language']
                )
                st.session_state.transcription = transcription
                st.session_state.transcription_time = transcription_time
                input_text = transcription
        else:
            st.error("⚠️ Please provide voice input or text query")
            return
        
        if input_text:
            # Initialize RAG engine (fresh instance each time)
            with st.spinner("Initializing RAG engine..."):
                try:
                    rag_engine = create_rag_engine(
                        db_config=config['db_config'],
                        ollama_model=config['llm_model'],
                        n_tables=config['n_tables'],
                        n_examples=config['n_examples']
                    )
                except Exception as e:
                    st.error(f"❌ Failed to initialize RAG engine: {str(e)}")
                    return
            
            # Process query
            with st.spinner("Processing query..."):
                try:
                    result = rag_engine.process_query(input_text)
                    
                    st.session_state.sql_query = result.get('sql_query', '')
                    st.session_state.query_results = result.get('results', {})
                    st.session_state.debug_info = result.get('debug_info', {})
                    
                    if result['success']:
                        st.success("✅ Query processed successfully!")
                    else:
                        st.error(f"❌ Query failed: {result.get('error', 'Unknown error')}")
                    
                    rag_engine.close()
                    
                except Exception as e:
                    st.error(f"❌ Processing error: {str(e)}")
                    return
    
    # Display results
    st.markdown("---")
    render_results()
    
    st.markdown("---")
    render_debug_info(config['show_prompt'])
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6a6a6a; padding: 2rem;'>
        <p>VoxQuery - Voice-Powered SQL Generation with RAG</p>
        <p>Built with Streamlit • Whisper • LLaMA • ChromaDB</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()