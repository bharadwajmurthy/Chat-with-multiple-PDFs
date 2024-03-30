
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter=CharacterTextSplitter()
    text_chunks= text_splitter.split_text(text)
    return text_chunks

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Multiple pdfs",page_icon=":books:")
    
    st.header("Chat with Multiple pdfs")
    
    st.text_input("Ask a question about your documents")
    
    with st.sidebar:
        st.subheader("Your documents")
        
        pdf_docs=st.file_uploader("Upload your pdfs here and click on 'Process'",accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing your documents..."):
                
                # get the pdf text
                raw_text = get_pdf_text(pdf_docs)
                
                #get the text chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(raw_text)
                #create vector store



if __name__ == '__main__':
    main()