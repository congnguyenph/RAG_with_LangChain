import os
from dotenv import load_dotenv
import streamlit as st
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader, UnstructuredPowerPointLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

#Load bến môi trường
load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def load_document(file_path):
    # Lấy phần mở rộng của tệp
    file_extension = os.path.splitext(file_path)[1].lower()

    # Kiểm tra loại tệp và sử dụng pp loader phù hợp
    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path)

    elif file_extension == ".doc" or file_extension == ".docx":
        loader = Docx2txtLoader(file_path)

    elif file_extension == ".xls" or file_extension == ".xlsx":
        loader = UnstructuredExcelLoader(file_path)

    elif file_extension == ".ppt" or file_extension == ".pptx":
        loader = UnstructuredPowerPointLoader(file_path)

    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    documents = loader.load()
    return documents
def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(
        # separator="/n",
        chunk_size=1200,
        chunk_overlap=200
    )
    doc_chunks = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    return vectorstore

def create_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="map_reduce",
        memory=memory,
        verbose=True
    )
    return chain



st.title("RAG Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_files = st.file_uploader(
    label="Upload your pdf files",
    type=["pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx"],
    accept_multiple_files=True
)
# Xử lý nếu có tệp được tải lên
if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        file_path = f"{working_dir}/upload/{uploaded_file.name}"

        if os.path.exists(file_path):#xóa tệp nếu nó tồn tại
            os.remove(file_path)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Đọc nội dung của từng tệp PDF và thêm vào danh sách documents
        documents.extend(load_document(file_path))

    # Thiết lập vectorstore từ tất cả tài liệu đã tải lên nếu chưa thiết lập
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = setup_vectorstore(documents)

    # Thiết lập conversation chain nếu chưa thiết lập
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_input = st.chat_input("Ask Llama...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = st.session_state.conversation_chain({"question": user_input})
        assistant_response = response["answer"]
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
