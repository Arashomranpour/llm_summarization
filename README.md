<div align="center">

# 📚 LLM PDF Summarization

**Upload a PDF and get a concise summary generated locally by the LaMini-Flan-T5 model from Hugging Face.**

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗_Transformers-FFD21E)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

</div>

---

## ✨ Features

- 📤 Upload a PDF in the browser.
- 👀 Side-by-side view: the original document and the generated summary.
- 🧠 Summarization with **`MBZUAI/LaMini-Flan-T5-248M`** through the Hugging Face `summarization` pipeline.
- ✂️ PDF text is loaded with LangChain's `PyPDFLoader` and split into chunks before summarizing.

## 🚀 Getting Started

```bash
git clone https://github.com/Arashomranpour/llm_summarization.git
cd llm_summarization
pip install streamlit transformers torch langchain pypdf sentencepiece
streamlit run "app (3).py"
```

The model is downloaded from Hugging Face on the first run.

## 📁 Project Structure

```
.
├── app (3).py    # Streamlit app + summarization pipeline
└── README.md
```

## 🛠️ Tech Stack

`Streamlit` · `Hugging Face Transformers` · `PyTorch` · `LangChain` · `LaMini-Flan-T5`
