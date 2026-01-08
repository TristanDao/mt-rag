import re
import streamlit as st
from typing import List, Dict

from Products_RAG_main.rag import RAGSystem
from Products_RAG_main.rerank import Reranker
from Products_RAG_main.reflection import Reflection


# =========================================================
# COLLECTION CONFIG
# =========================================================
COLLECTION_MAPPING = {
    "clapnq": "mt-rag-clapnq-elser-512-100-20240503",
    "govt": "mt-rag-govt-elser-512-100-20240611",
    "fiqa": "mt-rag-fiqa-beir-elser-512-100-20240501",
    "cloud": "mt-rag-ibmcloud-elser-512-100-20240502",
}


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="RAG Chatbot",
    layout="wide",
)


# =========================================================
# SIDEBAR – CONFIG
# =========================================================
st.sidebar.title("⚙️ Configuration")

use_vector_db = st.sidebar.checkbox(
    "Enable Retrieval (Qdrant)",
    value=True
)

top_k = st.sidebar.slider(
    "Final Top-K documents",
    min_value=1,
    max_value=10,
    value=5,
    disabled=not use_vector_db
)

per_collection_k = st.sidebar.slider(
    "Per-collection retrieve K",
    min_value=3,
    max_value=15,
    value=8,
    disabled=not use_vector_db
)


# =========================================================
# INIT RAG (CACHED)
# =========================================================
@st.cache_resource
def init_rag(use_vector_db: bool) -> RAGSystem:
    return RAGSystem(use_vector_db=use_vector_db)


rag = init_rag(use_vector_db)

reflection = Reflection(
    llm_client=rag.llm_client,
    llm_model=rag.llm_model
)

reranker = Reranker() if use_vector_db else None


# =========================================================
# SESSION STATE
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_state" not in st.session_state:
    st.session_state.conversation_state = {
        "intent": None,     # ask_solution / critique / specify_requirement / continue
        "stance": None,     # neutral / negative / positive
    }

if "user_facts" not in st.session_state:
    st.session_state.user_facts = {
        "name": None
    }


# =========================================================
# UTTERANCE INTERPRETER (INTENT / STANCE)
# =========================================================
def interpret_utterance(text: str) -> Dict:
    t = text.lower()

    if any(k in t for k in [
        "không tối ưu",
        "chưa ổn",
        "sai",
        "không đúng",
        "không hợp lý",
    ]):
        return {"intent": "critique", "stance": "negative"}

    if any(k in t for k in [
        "tôi muốn",
        "mục tiêu",
        "yêu cầu",
        "tôi cần",
    ]):
        return {"intent": "specify_requirement", "stance": "neutral"}

    if any(k in t for k in [
        "làm sao",
        "như thế nào",
        "cách",
        "phải làm gì",
    ]):
        return {"intent": "ask_solution", "stance": "neutral"}

    return {"intent": "continue", "stance": "neutral"}


# =========================================================
# FACT EXTRACTION (NAME ONLY – EXPLICIT FACT)
# =========================================================
def extract_user_facts(text: str):
    """
    Extract explicit, high-confidence user facts.
    This runs BEFORE Reflection.
    """
    match = re.search(
        r"(tôi\s+(tên|là))\s+([A-Za-zÀ-ỹ]+)",
        text,
        re.IGNORECASE
    )
    if match:
        st.session_state.user_facts["name"] = match.group(3)


# =========================================================
# HEADER
# =========================================================
st.title("RAG Chatbot")


# =========================================================
# CHAT HISTORY
# =========================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# =========================================================
# USER INPUT
# =========================================================
user_input = st.chat_input("Ask something...")

if user_input:
    # =====================================================
    # 0️⃣ FACT EXTRACTION (MUST RUN FIRST)
    # =====================================================
    extract_user_facts(user_input)

    # =====================================================
    # 1️⃣ INTERPRET UTTERANCE → UPDATE STATE
    # =====================================================
    interpretation = interpret_utterance(user_input)
    st.session_state.conversation_state["intent"] = interpretation["intent"]
    st.session_state.conversation_state["stance"] = interpretation["stance"]

    # =====================================================
    # 2️⃣ REFLECTION (STATELESS REWRITE)
    # =====================================================
    history = st.session_state.messages.copy()

    rewritten_query = reflection.rewrite(
        messages=history,
        current_query=user_input
    )

    # =====================================================
    # 3️⃣ APPEND USER MESSAGE
    # =====================================================
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    if rewritten_query != user_input:
        st.caption(f"🔎 Interpreted as: **{rewritten_query}**")

    # =====================================================
    # 4️⃣ MULTI-COLLECTION RETRIEVAL
    # =====================================================
    all_docs: List[Dict] = []

    if use_vector_db:
        for key, collection in COLLECTION_MAPPING.items():
            rag.collection_name = collection
            docs = rag.retrieve(rewritten_query, top_k=per_collection_k)

            for d in docs:
                d["source_collection"] = key

            all_docs.extend(docs)

    # =====================================================
    # 5️⃣ RERANK
    # =====================================================
    reranked_docs = (
        reranker.rerank(
            query=rewritten_query,
            documents=all_docs,
            top_k=top_k
        )
        if reranker and all_docs
        else all_docs[:top_k]
    )

    # =====================================================
    # 6️⃣ FORMAT CONTEXT
    # =====================================================
    context = rag.format_context(reranked_docs)

    # =====================================================
    # 7️⃣ ANSWER GENERATION (STATE + FACT AWARE)
    # =====================================================
    intent = st.session_state.conversation_state["intent"]
    stance = st.session_state.conversation_state["stance"]
    name = st.session_state.user_facts.get("name")

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = rag.response_generator.generate(
                prompt_type="standard" if use_vector_db and context else "free",
                query=f"""
User facts:
- Name: {name}

Conversation intent: {intent}
User stance: {stance}

User query:
{rewritten_query}
""".strip(),
                context=context if use_vector_db else ""
            )
            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    # =====================================================
    # 8️⃣ DOCUMENTS PANEL
    # =====================================================
    if reranked_docs:
        with st.expander("📄 Retrieved & Reranked Documents", expanded=False):
            for i, doc in enumerate(reranked_docs, start=1):
                st.markdown(f"**Document {i}**")

                if "source_collection" in doc:
                    st.caption(f"Collection: {doc['source_collection']}")

                payload = doc.get("payload", {})
                text = payload.get("text", "") or payload.get("content", "")

                if text:
                    st.markdown(text)
                else:
                    st.caption("(No text in payload)")

                st.markdown("---")
