
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI, ChatHuggingFace

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

# create vector store using OPENAI embeddings and HuGGINGFACE embeddings
def get_vector_store(text_chunks):
    # embeddings=OpenAIEmbeddings()
    embeddings=HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vector_store=FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    return vector_store

def get_conversation_chain(vector_store):
    llm=ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    conversation_chain=ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory  
    )
    return conversation_chain


def main():
    load_dotenv()
    
    if "conversation" not in st.session_state:
        st.session_state.conversation=None
    
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
                
                #create vector store
                vector_store = get_vector_store(text_chunks)

                st.session_state.conversation=get_conversation_chain(vector_store)


if __name__ == '__main__':
    main()