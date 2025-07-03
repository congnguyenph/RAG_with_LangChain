import os
from dotenv import load_dotenv
import streamlit as st

from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

# Load biến môi trường
load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ====================
# 1. Đăng nhập admin
# ====================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🎓 Chatbot Công tác Sinh viên")

# ================
# 2. Form đăng nhập
# ================
with st.expander("🔐 Đăng nhập Admin để tải tài liệu"):
    with st.form("login_form"):
        username = st.text_input("Tên đăng nhập")
        password = st.text_input("Mật khẩu", type="password")
        submit = st.form_submit_button("Đăng nhập")

        if submit:
            if username == "admin" and password == "admin123":
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("✅ Đăng nhập thành công!")
            else:
                st.error("❌ Sai tên đăng nhập hoặc mật khẩu.")

# =========================
# 3. Hàm xử lý tài liệu và vectorstore
# =========================

def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in [".doc", ".docx"]:
        loader = Docx2txtLoader(file_path)
    elif ext in [".xls", ".xlsx"]:
        loader = UnstructuredExcelLoader(file_path)
    elif ext in [".ppt", ".pptx"]:
        loader = UnstructuredPowerPointLoader(file_path)
    else:
        raise ValueError(f"Unsupported format: {ext}")
    return loader.load()

def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    return FAISS.from_documents(chunks, embeddings)

def create_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        chain_type="map_reduce",
        verbose=True
    )

# =========================
# 4. Nếu là Admin: Cho phép Upload file
# =========================

if st.session_state.logged_in and st.session_state.username == "admin":
    uploaded_files = st.file_uploader(
        label="📁 Chọn file để tải lên (chỉ dành cho admin)",
        type=["pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx"],
        accept_multiple_files=True
    )

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            upload_dir = os.path.join(working_dir, "upload")
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, uploaded_file.name)

            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except PermissionError:
                st.warning(f"⚠️ Không thể ghi đè file `{uploaded_file.name}`. Hãy đóng mọi chương trình đang mở file này.")
                continue

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success(f"✅ File `{uploaded_file.name}` đã được tải lên thành công!")

            documents.extend(load_document(file_path))

        # Tạo vectorstore và chain
        st.session_state.vectorstore = setup_vectorstore(documents)
        st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)
        st.success("✅ Đã nạp toàn bộ tài liệu thành công!")


# =========================
# 5. Hiển thị lịch sử chat
# =========================
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================
# 6. Chat input
# =========================
user_input = st.chat_input("Hỏi chatbot về công tác sinh viên...")

if user_input:
    if "conversation_chain" not in st.session_state:
        st.error("⚠️ Chưa có dữ liệu để trả lời. Admin cần upload tài liệu trước.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            response = st.session_state.conversation_chain({"question": user_input})
            assistant_response = response["answer"]
            st.markdown(assistant_response)

            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
