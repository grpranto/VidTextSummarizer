# Updated Documentation:
# 1. This script builds a Streamlit application for summarizing content from a YouTube video or a website.
# 2. Dependencies include Streamlit, LangChain, and related libraries for natural language processing and integration with HuggingFace.
# 3. The user inputs their HuggingFace API token and a URL (either a YouTube video or a webpage). The token is required for accessing the HuggingFace model.
# 4. The application validates the input. If the URL is for a YouTube video, the corresponding loader extracts text from the video's subtitles. For general websites, an unstructured loader fetches and processes the page content.
# 5. A customizable prompt template guides the summarization process, ensuring concise and informative summaries tailored to the input text.
# 6. The HuggingFaceEndpoint connects to the mistralai/Mistral-7B-Instruct-v0.3, which generates the summary. Users can configure this to use a different HuggingFace model if needed.
# 7. The application includes exception handling to manage errors, such as invalid URLs or API token issues, gracefully.

# Notes:
# - Ensure the HuggingFace API token is valid and has access to the specified model repository.
# - The mistralai/Mistral-7B-Instruct-v0.3 model is chosen for its summarization capabilities. Other models can be substituted as needed.

import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
import nltk
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq

# Download NLTK data required for text processing
nltk.download('averaged_perceptron_tagger_eng')

# Streamlit App Configuration
st.set_page_config(
    page_title="Text Summarization from Website or Youtube Video URL",
    page_icon="📝"
)

# Application Title and Subtitle
st.title("📝 Text Summarization from Website or Youtube Video URL")
st.subheader("Summarize URL")

# Sidebar: Get the Huggingface API Token
with st.sidebar:
    hf_api_key = st.text_input(
        "Huggingface API Token",
        value="",
        type="password",
    )

# Input field for the URL to be summarized
generic_url = st.text_input(
    "URL",
    label_visibility="collapsed",
    placeholder="e.g. https://python.langchain.com/docs/introduction/"
)

# Prompt Template for Summarization
prompt_template = """
Provide the summary of the following content in 300 words:
{text}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["text"]
)

# Summarization Logic
if st.button("Summarize"):
    # validate inputs
    if not hf_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please provide a valid URL. It should be YT video or website URL")
    else:
        try:
            with st.spinner("Waiting..."):
                # Load content from the URL (YouTube or Website)
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(
                        generic_url
                    )
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    )

                docs = loader.load()

                # # Configure HuggingFace LLM Endpoint
                # repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
                # llm = HuggingFaceEndpoint(
                #     repo_id=repo_id,
                #     max_new_tokens=150,
                #     temperature=0.7,
                #     huggingfacehub_api_token=hf_api_key
                # )
                llm = ChatGroq(
                    model="gemma2-9b-it",
                    groq_api_key=hf_api_key
                )

                # Summarization Chain
                chain = load_summarize_chain(
                    llm=llm,
                    chain_type="stuff",
                    prompt=prompt
                )

                # Generate the Summary
                output_summary = chain.invoke(docs)
                summary_text = output_summary.get('output_text', '').strip()

                # Display the Summary
                st.success(summary_text)

        except Exception as e:
            st.exception(f"Exception:{e}")
