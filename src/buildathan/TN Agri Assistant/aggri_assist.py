"""
Streamlit front end for the PDF RAG system (OpenAI + LangChain + FAISS).

- PDFs are read from the ./pdfs folder and indexed automatically by this code.
  If the PDFs change (added, removed or modified), the index is rebuilt.
- The sidebar keeps a search history (saved in search_history.json), so it survives restarts.

Run:
    pip install streamlit langchain langchain-community langchain-openai \
                langchain-text-splitters pypdf faiss-cpu python-dotenv
    export OPENAI_API_KEY="sk-..."      # or put it in a .env file
    mkdir pdfs  &&  copy your PDFs into it
    streamlit run streamlit_app.py
"""

import base64
import glob
import hashlib
import json
import os
import shutil
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ---------- config ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "assets", "Tamil_Nadu_emb.png")
PDF_DIR = os.path.join(BASE_DIR, "pdfs")
INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")
FINGERPRINT_FILE = os.path.join(INDEX_DIR, "pdf_fingerprint.txt")
HISTORY_FILE = os.path.join(BASE_DIR, "search_history.json")
MAX_HISTORY = 100
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4

SYSTEM_PROMPT = """You are an assistant that helps farmers and the public with \
agricultural schemes of the Government of Tamil Nadu (subsidies, crop insurance, \
irrigation, horticulture, farm mechanisation, loans, procurement, and similar programmes).

Rules:
1. Answer ONLY from the context below, which comes from official scheme documents. \
Never use outside knowledge and never invent scheme names, amounts, dates, \
eligibility rules or deadlines.
2. If the question is not about Tamil Nadu agricultural schemes, politely say you can \
only help with Tamil Nadu agricultural schemes, and suggest a relevant question.
3. If the context does not contain the answer, say it is not covered in the available \
documents and suggest checking with the local Agriculture / Horticulture office or the \
official Tamil Nadu agriculture website.
4. When a scheme is described, organise the answer under these headings where the \
information is available: Scheme, Benefits, Eligibility, How to apply, Documents required.
5. Keep the language simple. Reply in the same language the user wrote in (Tamil or English).
6. Cite sources as [file p.N].

Context:
{context}"""

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "{question}")]
)

st.set_page_config(
    page_title="Tamil Nadu Agricultural Schemes Assistant",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🌾",
    layout="centered",
)

# Custom styling: Forest green sidebar & Light green chat window screen
st.markdown(
    """
    <style>
    /* Main chat window screen: light green */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #f0f7f0 !important;
    }

    /* High-contrast text readability across main screen */
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] span,
    [data-testid="stAppViewContainer"] label,
    [data-testid="stAppViewContainer"] li {
        color: #0b2917 !important;
        font-weight: 500;
    }

    /* High-contrast for all captions and metadata */
    [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"],
    [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] p,
    [data-testid="stAppViewContainer"] small {
        color: #11381e !important;
        font-size: 0.96rem !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }

    /* Bottom area / chat input bar */
    [data-testid="stBottom"], [data-testid="stBottom"] > div {
        background-color: #f0f7f0 !important;
    }

    /* Sidebar: forest green */
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
        background-color: #1b4332 !important;
    }

    /* Sidebar typography & contrast */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown {
        color: #f0fdf4 !important;
    }

    /* Sidebar captions and timestamps */
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] small {
        color: #d8f3dc !important;
    }

    /* Sidebar divider */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.25) !important;
    }

    /* Sidebar chat history buttons */
    [data-testid="stSidebar"] button {
        background-color: rgba(255, 255, 255, 0.12) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        border-radius: 8px !important;
        transition: all 0.2s ease-in-out;
    }

    [data-testid="stSidebar"] button:hover {
        background-color: rgba(255, 255, 255, 0.24) !important;
        border-color: rgba(255, 255, 255, 0.5) !important;
        color: #ffffff !important;
    }

    /* Clear history button distinct styling */
    [data-testid="stSidebar"] div.st-key-clear_history_btn > button {
        background-color: rgba(220, 38, 38, 0.18) !important;
        color: #fecaca !important;
        border: 1px solid rgba(239, 68, 68, 0.35) !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease-in-out !important;
    }

    [data-testid="stSidebar"] div.st-key-clear_history_btn > button:hover {
        background-color: rgba(220, 38, 38, 0.32) !important;
        border-color: rgba(239, 68, 68, 0.65) !important;
        color: #ffffff !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15) !important;
    }

    /* Chat message cards */
    [data-testid="stChatMessage"] {
        background-color: #ffffff !important;
        border: 1px solid #d0e7d2 !important;
        border-radius: 12px !important;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
        margin-bottom: 0.75rem;
    }

    /* Sources expander on light green background */
    [data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1px solid #d0e7d2 !important;
        border-radius: 8px !important;
    }

    /* Chat input styling - clean white surface, removing forest green */
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] > div > div,
    [data-testid="stChatInput"] div[data-baseweb="base-input"] {
        background-color: #ffffff !important;
        border-color: #b7dfb9 !important;
        border-radius: 12px !important;
    }

    [data-testid="stChatInput"]:focus-within,
    [data-testid="stChatInput"] div[data-baseweb="base-input"]:focus-within {
        border-color: #2d6a4f !important;
        box-shadow: 0 0 0 2px rgba(45, 106, 79, 0.2) !important;
    }

    [data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        color: #1e392a !important;
    }

    /* Send button inside chat input */
    [data-testid="stChatInput"] button {
        background-color: #2d6a4f !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
    }

    [data-testid="stChatInput"] button svg {
        fill: #ffffff !important;
        stroke: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ---------- indexing (runs inside the code) ----------
def list_pdfs() -> list[str]:
    return sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))


def pdf_fingerprint(paths: list[str]) -> str:
    """Hash of file names, sizes and modified times: changes when the PDFs change."""
    h = hashlib.sha256()
    for p in paths:
        s = os.stat(p)
        h.update(f"{os.path.basename(p)}|{s.st_size}|{int(s.st_mtime)}".encode())
    return h.hexdigest()


def index_is_current(fingerprint: str) -> bool:
    if not (os.path.isdir(INDEX_DIR) and os.path.isfile(FINGERPRINT_FILE)):
        return False
    with open(FINGERPRINT_FILE) as f:
        return f.read().strip() == fingerprint


def build_index(paths: list[str], embeddings: OpenAIEmbeddings) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = []
    for path in paths:
        pages = PyPDFLoader(path).load()  # one Document per page
        chunks.extend(splitter.split_documents(pages))

    if not chunks:
        raise ValueError("No text could be extracted from the PDFs (scanned images?).")

    vs = FAISS.from_documents(chunks, embeddings)

    shutil.rmtree(INDEX_DIR, ignore_errors=True)  # rebuild from scratch: no duplicates
    vs.save_local(INDEX_DIR)
    with open(FINGERPRINT_FILE, "w") as f:
        f.write(pdf_fingerprint(paths))
    return vs


@st.cache_resource(show_spinner="Indexing PDFs from the pdfs folder...")
def get_vectorstore() -> FAISS:
    """Load the FAISS index, (re)building it first if the PDFs changed. Cached per session."""
    paths = list_pdfs()
    if not paths:
        raise FileNotFoundError(f"No PDF files found in ./{PDF_DIR}")

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    if index_is_current(pdf_fingerprint(paths)):
        return FAISS.load_local(
            INDEX_DIR, embeddings, allow_dangerous_deserialization=True
        )
    return build_index(paths, embeddings)


# ---------- question answering ----------
def format_docs(docs) -> str:
    return "\n\n".join(
        f"[{os.path.basename(d.metadata.get('source', '?'))} "
        f"p.{d.metadata.get('page', 0) + 1}]\n{d.page_content}"
        for d in docs
    )


def answer(question: str):
    docs = get_vectorstore().as_retriever(search_kwargs={"k": TOP_K}).invoke(question)
    chain = PROMPT | ChatOpenAI(model=CHAT_MODEL, temperature=0) | StrOutputParser()
    text = chain.invoke({"context": format_docs(docs), "question": question})
    sources = [
        {
            "label": f"{os.path.basename(d.metadata.get('source', '?'))} "
                     f"- page {d.metadata.get('page', 0) + 1}",
            "text": d.page_content,
        }
        for d in docs
    ]
    return text, sources


def show_sources(sources) -> None:
    with st.expander("Sources"):
        for s in sources:
            st.markdown(f"**{s['label']}**")
            st.text(s["text"])


# ---------- search history (saved to disk) ----------
def load_history() -> list[dict]:
    if os.path.isfile(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
    return []


def save_history(history: list[dict]) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def add_to_history(question: str, reply: str, sources: list[dict]) -> None:
    history = st.session_state.history
    history.append(
        {
            "question": question,
            "answer": reply,
            "sources": sources,
            "time": datetime.now().strftime("%d %b %Y, %H:%M"),
        }
    )
    st.session_state.history = history[-MAX_HISTORY:]
    save_history(st.session_state.history)


def open_entry(i: int) -> None:
    st.session_state.selected = i


def back_to_chat() -> None:
    st.session_state.selected = None


def clear_history() -> None:
    st.session_state.history = []
    st.session_state.selected = None
    save_history([])


# ---------- state ----------
if "history" not in st.session_state:
    st.session_state.history = load_history()
st.session_state.setdefault("selected", None)
st.session_state.setdefault("messages", [])

os.makedirs(PDF_DIR, exist_ok=True)
pdf_files = list_pdfs()

# ---------- sidebar: search history & info ----------
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        sidebar_logo_b64 = get_image_base64(LOGO_PATH)
        st.markdown(
            f"""
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; margin-bottom: 1rem; width: 100%;">
                <img src="data:image/png;base64,{sidebar_logo_b64}" width="85" alt="Tamil Nadu Government Logo" style="display: block; margin: 0 auto 10px auto;" />
                <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff; line-height: 1.3; text-align: center;">தமிழ்நாடு அரசு</div>
                <div style="font-size: 0.9rem; font-weight: 500; color: #d8f3dc; font-style: italic; margin-top: 3px; text-align: center;">Government of Tamil Nadu</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()
    st.header("Chat history")
    history = st.session_state.history
    if not history:
        st.caption("No chat history yet.")
    for i in reversed(range(len(history))):
        q = history[i]["question"]
        label = q if len(q) <= 42 else q[:39] + "..."
        st.button(
            label,
            key=f"hist_{i}",
            on_click=open_entry,
            args=(i,),
            help=f"{q}\n\n{history[i]['time']}",
            use_container_width=True,
        )
    if history:
        st.divider()
        st.button(
            "Clear history",
            key="clear_history_btn",
            on_click=clear_history,
            icon=":material/delete_outline:",
            use_container_width=True,
        )

    if pdf_files:
        st.divider()
        files_str = ", ".join(os.path.basename(p) for p in pdf_files)
        st.markdown(
            f"""
            <div style="margin-top: 1rem; padding: 10px 12px; background: rgba(255, 255, 255, 0.08); border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.16); font-size: 0.78rem; line-height: 1.45;">
                <div style="color: #ffffff; font-weight: 600; margin-bottom: 4px;">ℹ️ Indexed {len(pdf_files)} PDF(s)</div>
                <div style="color: #d8f3dc; opacity: 0.92; word-break: break-word;">{files_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------- main page ----------
if os.path.exists(LOGO_PATH):
    logo_b64 = get_image_base64(LOGO_PATH)
    st.markdown(
        f"""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; width: 100%; margin: -1rem auto 1.5rem auto;">
            <img src="data:image/png;base64,{logo_b64}" width="145" alt="Tamil Nadu Government Logo" style="display: block; margin: 0 auto 14px auto;" />
            <h1 style="color: #072110; font-size: 2.35rem; font-weight: 800; margin: 0; padding: 0; letter-spacing: -0.5px; text-align: center; width: 100%;">Tamil Nadu Agricultural Schemes Assistant</h1>
            <p style="color: #0d3419; font-size: 1.15rem; font-weight: 700; margin: 8px 0 4px 0; text-align: center; width: 100%;">Government of Tamil Nadu | தமிழ்நாடு அரசு • உழவர் நலத்துறை</p>
            <p style="color: #143f21; font-size: 1.02rem; font-weight: 600; margin: 0; text-align: center; width: 100%;">Ask about subsidies, crop insurance, irrigation, loans and other state schemes.</p>
            <p style="color: #0d3419; font-size: 1.0rem; font-weight: 700; margin: 4px 0 0 0; text-align: center; width: 100%;">தமிழிலும் கேட்கலாம்.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.title("🌾 Tamil Nadu Agricultural Schemes Assistant")
    st.caption("Ask about subsidies, crop insurance, irrigation, loans and other state schemes.\n\nதமிழிலும் கேட்கலாம்.")

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set. Add it to your environment or a .env file, then restart.")
    st.stop()

if not pdf_files:
    st.warning(
        f"Put the Tamil Nadu agricultural scheme PDFs (government orders, scheme "
        f"guidelines, brochures) in the **{PDF_DIR}/** folder, then refresh this page."
    )
    st.stop()

try:
    get_vectorstore()  # builds the index on first run
except Exception as e:
    st.error(f"Indexing failed: {e}")
    st.stop()

selected = st.session_state.selected
if selected is not None and selected < len(st.session_state.history):
    # Viewing a saved search from the sidebar
    entry = st.session_state.history[selected]
    st.button("Back to chat", on_click=back_to_chat, icon=":material/arrow_back:")
    st.caption(f"Saved search from {entry['time']}")
    with st.chat_message("user"):
        st.markdown(entry["question"])
    with st.chat_message("assistant"):
        st.markdown(entry["answer"])
        if entry.get("sources"):
            show_sources(entry["sources"])
else:
    # Current conversation
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                show_sources(m["sources"])

question = st.chat_input("Ask about a Tamil Nadu agricultural scheme, e.g. subsidy for drip irrigation")

if question:
    st.session_state.selected = None
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply, sources = answer(question)
                failed = False
            except Exception as e:
                reply, sources, failed = f"Error: {e}", [], True

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "sources": sources}
    )
    if not failed:
        add_to_history(question, reply, sources)
    st.rerun()  # redraw so the sidebar shows the new history entry
