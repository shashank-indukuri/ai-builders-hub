import streamlit as st
import assemblyai as aai
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the app
st.set_page_config(
    page_title="AI Subtitle Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for beautiful styling (dark theme compatible)
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 3rem 2rem;
    border-radius: 20px;
    color: white;
    text-align: center;
    margin-bottom: 3rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}
.upload-section {
    background: var(--background-color, #f8f9fa);
    padding: 2rem;
    border-radius: 15px;
    border: 2px dashed #667eea;
    text-align: center;
    margin: 2rem 0;
    color: var(--text-color, #333);
}
.feature-card {
    background: var(--secondary-background-color, #f8f9fa);
    padding: 2rem;
    border-radius: 15px;
    margin: 1rem 0;
    text-align: center;
    color: var(--text-color, #333);
    border: 1px solid var(--border-color, #e0e0e0);
}

/* Dark theme support */
@media (prefers-color-scheme: dark) {
    .upload-section {
        background: #262730;
        color: #fafafa;
        border-color: #667eea;
    }
    .feature-card {
        background: #262730;
        color: #fafafa;
        border-color: #404040;
    }

}
</style>
""", unsafe_allow_html=True)

# Beautiful header
st.markdown("""
<div class="main-header">
    <h1>🎬 AI Subtitle Generator</h1>
    <h3>Transform any audio into professional subtitles instantly</h3>
    <p>Upload → Process → Download • It's that simple!</p>
</div>
""", unsafe_allow_html=True)

# Initialize AssemblyAI
api_key = os.getenv("ASSEMBLYAI_API_KEY")
if not api_key:
    st.error("⚠️ Please set ASSEMBLYAI_API_KEY in your .env file")
    st.stop()
aai.settings.api_key = api_key

# Upload section
st.markdown("### 📤 Drop your audio or video file here")
uploaded_file = st.file_uploader(
    "",
    type=['mp3', 'wav', 'mp4', 'mov', 'avi', 'm4a', 'flac'],
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file:
    # File info and preview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.success(f"✅ **{uploaded_file.name}** ({uploaded_file.size / 1024 / 1024:.1f} MB)")
        st.audio(uploaded_file)
    
    with col2:
        speaker_detection = st.toggle("🎤 Speaker Detection", value=True)
        export_format = st.selectbox(
            "📥 Format",
            ["SRT", "VTT", "TXT"],
            label_visibility="collapsed"
        )
    
    # Generate button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🎆 Generate Subtitles", type="primary", use_container_width=True):
        with st.spinner("🤖 AI is processing your audio..."):
            try:
                temp_file = f"temp_{uploaded_file.name}"
                with open(temp_file, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                transcriber = aai.Transcriber()
                config = aai.TranscriptionConfig(speaker_labels=speaker_detection)
                transcript = transcriber.transcribe(temp_file, config=config)
                os.remove(temp_file)
                
                st.success("🎉 Subtitles generated successfully!")
                
                # Results section
                st.markdown("### 📝 Results")
                result_col1, result_col2 = st.columns([3, 1])
                
                with result_col1:
                    if export_format == "SRT":
                        content = transcript.export_subtitles_srt()
                        st.code(content[:1500] + "..." if len(content) > 1500 else content, language="")
                        st.download_button(
                            "📥 Download SRT",
                            data=content,
                            file_name=f"subtitles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.srt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    elif export_format == "VTT":
                        content = transcript.export_subtitles_vtt()
                        st.code(content[:1500] + "..." if len(content) > 1500 else content, language="")
                        st.download_button(
                            "📥 Download VTT",
                            data=content,
                            file_name=f"subtitles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.vtt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    else:
                        st.text_area("", transcript.text, height=400, label_visibility="collapsed")
                        st.download_button(
                            "📥 Download TXT",
                            data=transcript.text,
                            file_name=f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                
                with result_col2:
                    st.metric("📊 Words", len(transcript.text.split()))
                    
                    sentences = transcript.get_sentences()
                    if sentences:
                        duration = (sentences[-1].end - sentences[0].start) / 1000
                        st.metric("⏱️ Duration", f"{duration:.1f}s")
                        st.metric("📝 Sentences", len(sentences))
                    
                    if speaker_detection and hasattr(transcript, 'utterances'):
                        speakers = set()
                        for utterance in transcript.utterances:
                            if utterance.speaker:
                                speakers.add(utterance.speaker)
                        if speakers:
                            st.metric("🎤 Speakers", len(speakers))
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Try a smaller file or check your connection")

else:
    # Feature showcase
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h2>🎤</h2>
            <h4>Speaker Detection</h4>
            <p>Identifies who said what</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h2>⚡</h2>
            <h4>Lightning Fast</h4>
            <p>Results in under 60 seconds</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h2>🎯</h2>
            <h4>Professional Quality</h4>
            <p>AI-powered accuracy</p>
        </div>
        """, unsafe_allow_html=True)
st.markdown("*Built with Streamlit and AssemblyAI*")
