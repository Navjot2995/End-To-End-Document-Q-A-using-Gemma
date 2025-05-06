# 📚 Gemma PDF Analyzer

![Demo Animation](https://lottie.host/f1216edb-4e09-46e5-8f1e-90c367b6fc13/iM4N0EXuvy.json)

A powerful AI-powered document analysis tool that lets you chat with your PDFs using Google's Gemma model through Groq's ultra-fast inference engine.

## ✨ Features

- **Document Processing**: Extract and analyze text from multiple PDFs
- **AI Chat Interface**: Ask questions about your uploaded documents
- **Vector Search**: FAISS-based semantic document retrieval
- **Beautiful UI**: Customizable light/dark theme with animations
- **Blazing Fast**: Powered by Groq's LPU inference engine
- **Markdown Support**: Formatted responses with code highlighting

## 🛠️ Tech Stack

| Component            | Technology |
|----------------------|------------|
| Large Language Model | Gemma-2-9b-it (via Groq) |
| Embeddings           | Google Gemini (models/embedding-001) |
| Vector Store         | FAISS |
| Framework            | LangChain |
| Frontend             | Streamlit |
| PDF Processing       | PyPDFLoader |

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- [Groq API key](https://console.groq.com/)
- [Google API key](https://aistudio.google.com/)

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/gemma-pdf-analyzer.git
cd gemma-pdf-analyzer