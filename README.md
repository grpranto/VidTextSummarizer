# YouTube & Web Content Summarizer

This is a **Streamlit-based application** that extracts and summarizes content from a YouTube video or a webpage using the **DeepSeek-R1-Distill-Qwen-32B** model from HuggingFace.

## Features
- **Summarizes content** from YouTube videos (via subtitles) and websites.
- Uses **DeepSeek-R1-Distill-Qwen-32B** for high-quality summarization.
- **Customizable prompt template** ensures summaries are informative and well-structured.
- **Supports HuggingFace API integration**, allowing users to configure different models.
- **Handles errors gracefully**, ensuring smooth user experience.

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/VidTextSummarizer.git
cd VidTextSummarizer
```

### 2. Set Up the Virtual Environment

Using Conda:
```bash
conda create -p venv python==3.10 -y
conda activate ./venv
```

Using venv (alternative):
```bash
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate    # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### 1. Start the Streamlit App
```bash
streamlit run app.py
```

### 2. Provide Inputs
- **HuggingFace API Token** (required)
- **URL** (YouTube video or a webpage)
- Click **Summarize** to generate a summary.

## Improvements for Better Output
1. **Enhanced Prompt Engineering**: Improved prompt ensures structured and meaningful summarization with complete sentences.
2. **Token Filtering**: Removed unwanted `<\think>` and other system tags from model output.
3. **Robust URL Handling**: Validates and processes both YouTube and general web URLs efficiently.
4. **Configurable HuggingFace Model**: Users can modify the model choice if needed.
5. **Better Error Handling**: Graceful error messages for invalid URLs, API issues, and content retrieval failures.

## Example Demo URLs
- 🌐 **Web**: [Wikipedia - DeepSeek](https://en.wikipedia.org/wiki/DeepSeek)
- 🎥 **YouTube**: [FIFA Video](https://www.youtube.com/watch?v=MCWJNOfJoSM&t=11s&ab_channel=FIFA)

## License
This project is licensed under the MIT License.

## Contributing
Feel free to submit pull requests or report issues to improve the project!

---

Developed with ❤️ using **Streamlit**, **LangChain**, and **HuggingFace**.