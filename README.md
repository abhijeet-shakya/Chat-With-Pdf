# Chat-With-PDF

A Streamlit-based application that allows you to chat with PDF documents using Google's Generative AI (Gemini). Upload PDFs, ask questions, and get detailed answers extracted from the document content.

## Features

- 📄 **PDF Upload**: Upload multiple PDF files simultaneously
- 🤖 **AI-Powered QA**: Ask questions about your PDFs using Google's Gemini AI
- 💾 **Vector Store**: Uses FAISS for efficient similarity search
- 🔄 **Chat History**: Maintains conversation history during your session
- 📝 **Context-Aware Responses**: Extracts relevant information from PDFs for accurate answers

## Prerequisites

- Python 3.8 or higher
- Google API Key (for Generative AI)

## Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd Chat-With-Pdf
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory and add your Google API key:

   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

   Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

Then:

1. Open your browser to the provided URL (usually `http://localhost:8501`)
2. Upload one or more PDF files
3. Ask questions about the content of your PDFs
4. Get detailed answers powered by Gemini AI

## Project Structure

```
Chat-With-Pdf/
├── app.py              # Main Streamlit application
├── app2.py             # Alternative app version
├── requirements.txt    # Python dependencies
├── test.py            # Test file
└── README.md          # This file
```

## Technologies Used

- **Streamlit**: Web framework for the UI
- **Google Generative AI**: Gemini AI for question answering
- **LangChain**: Framework for building AI applications
- **PyPDF2**: PDF file parsing
- **FAISS**: Vector store for similarity search
- **python-dotenv**: Environment variable management

## How It Works

1. **PDF Processing**: PDFs are read and text is extracted using PyPDF2
2. **Text Chunking**: Content is split into manageable chunks using RecursiveCharacterTextSplitter
3. **Vector Embeddings**: Text chunks are converted to embeddings using Google's embedding model
4. **Vector Store**: Embeddings are stored in FAISS for fast retrieval
5. **Question Answering**: User questions are processed, relevant chunks are retrieved, and Gemini generates answers

## Dependencies

See `requirements.txt` for all dependencies:

- streamlit
- google-generativeai
- python-dotenv
- langchain
- PyPDF2
- chromadb
- faiss-cpu
- langchain_google_genai

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Support

For issues and questions, please open an issue on the GitHub repository.
