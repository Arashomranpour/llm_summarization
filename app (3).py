import streamlit as st
from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from transformers import T5Tokenizer,T5ForConditionalGeneration
from transformers import pipeline
import base64
import torch


# model tokenizer
nam="MBZUAI/LaMini-Flan-T5-248M"
tok=T5Tokenizer.from_pretrained(nam)
base=T5ForConditionalGeneration.from_pretrained(nam,device_map="auto",torch_dtype=torch.float32)


def prepr(file):
  loader=PyPDFLoader(file)
  pages=loader.load_and_split()
  txt_splt=RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=50)
  text=txt_splt.split_documents(pages)
  latest=""
  for t in text:
    latest=latest+t.page_content

  return latest


# pipeline

def llm_pipe(filepath):
  pipe_sum=pipeline(
    "summarization",
    model=base,
    tokenizer=tok,
    max_lenght=500
    ,min_lenght=50
  )
  input_t=prepr(filepath)
  res=pipe_sum(input_t)
  res=res[0]["summary_text"]
  return res


@st.cache_data

def display(files):
  with open(files,"rb") as f:
    base64_pdf=base64.b64encode(f.read().decode())

    pdf_dis=F"<iframe src='data:application/pdf;base64,{base64_pdf}' width='100%' height='600' type='application/pdf'></iframe>"
    st.markdown(pdf_dis,unsafe_allow_html=True)


st.set_page_config(layout="wide",page_title="Summarization")

def main():
  st.title("Document text summarization")
  uploaded=st.file_uploader("upload your pdf",type=["pdf"])
  if uploaded is not None:
    if st.button("Proceed"):
      file_path="data/"+uploaded.name
      with open(file_path,"wb") as temp:
        temp.write(uploaded.read())

      col1,col2=st.columns(2)

      with col1:
        st.info("uploaded file")
        pdfv=display(file_path)

      with col2:
        st.info("Summarization is below")
        sums=llm_pipe(file_path)
        st.success(sums)

if __name__=="__main__":
  main()


