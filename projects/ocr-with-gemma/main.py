import streamlit as st
import ollama
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="OCR with Gemma",
    page_icon="📄",
    layout="wide"
)

# Header
st.title("📄 OCR with Gemma")
st.markdown("*Extract text from images using Ollama's Gemma Vision model*")

# Clear button
if st.button("🗑️ Clear Results"):
    if 'ocr_result' in st.session_state:
        del st.session_state['ocr_result']
    st.rerun()

st.markdown("---")

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['png', 'jpg', 'jpeg', 'webp'],
        help="Upload an image containing text to extract"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("🔍 Extract Text", type="primary"):
            with st.spinner("Extracting text from image..."):
                try:
                    response = ollama.chat(
                        model='gemma3:12b',
                        messages=[{
                            'role': 'user',
                            'content': """Carefully analyze this image and extract all visible text content. 
                                        Present the extracted text in a clean, structured format with proper 
                                        organization. Maintain the original hierarchy and formatting where possible 
                                        (headings, paragraphs, lists, tables). Ensure accuracy and readability 
                                        in the final output.""",
                            'images': [uploaded_file.getvalue()]
                        }]
                    )
                    st.session_state['ocr_result'] = response['message']['content']
                    st.success("Text extracted successfully!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Make sure Ollama is running and gemma3:12b model is installed.")

with col2:
    st.subheader("📝 Extracted Text")
    
    if 'ocr_result' in st.session_state:
        # Display results in a text area for easy copying
        st.text_area(
            "Extracted Content",
            value=st.session_state['ocr_result'],
            height=400,
            help="Copy the extracted text from here"
        )
        
        # Download button
        st.download_button(
            label="💾 Download as TXT",
            data=st.session_state['ocr_result'],
            file_name="extracted_text.txt",
            mime="text/plain"
        )
    else:
        st.info("👈 Upload an image and extract text to see results here")

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit and Ollama*")