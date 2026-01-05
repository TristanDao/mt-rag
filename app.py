import streamlit as st
from typing import List, Dict

from Products_RAG_main.rag import RAGSystem, RAGMode


# =========================================================
# REFLECTION (LLM-BASED QUERY CONTEXTUALIZATION)
# =========================================================
class Reflection:
    def __init__(self, llm_client, llm_model: str):
        self.llm_client = llm_client
        self.llm_model = llm_model

    def rewrite(self, messages: List[Dict], current_query: str) -> str:
        """
        Rewrite the current user query into a standalone question
        that fully captures the conversational context.
        """

        system_prompt = """You are a query rewriting assistant.

Your task is to rewrite the user's latest question into a standalone,
self-contained question that can be understood without the prior conversation.

Rules:
- Do NOT answer the question.
- Do NOT add new information.
- Preserve the user's original intent.
- If the question asks to elaborate (e.g. "nói thêm", "tell me more", "why"),
  expand it based on the immediately preceding topic.
- If the question is already self-contained, return it unchanged.
"""

        prompt_messages = [{"role": "system", "content": system_prompt}]

        # Use last 4–6 turns for context
        for msg in messages[-6:]:
            if msg["role"] in ("user", "assistant"):
                prompt_messages.append(
                    {
                        "role": msg["role"],
                        "content": msg["content"]
                    }
                )

        # Latest user query
        prompt_messages.append(
            {
                "role": "user",
                "content": current_query
            }
        )

        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=prompt_messages,
            temperature=0.0,
            max_tokens=120,
        )

        rewritten = response.choices[0].message.content.strip()

        # Fallback safety
        if not rewritten or len(rewritten) < 5:
            return current_query

        return rewritten


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
    "Top-K documents",
    min_value=1,
    max_value=10,
    value=5,
    disabled=not use_vector_db
)


# =========================================================
# INIT RAG (CACHED)
# =========================================================
@st.cache_resource
def init_rag(use_vector_db: bool) -> RAGSystem:
    return RAGSystem(use_vector_db=use_vector_db)


rag = init_rag(use_vector_db)

# Init Reflection (LLM-based)
reflection = Reflection(
    llm_client=rag.llm_client,
    llm_model=rag.llm_model
)


# =========================================================
# SESSION STATE (CHAT HISTORY)
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# HEADER
# =========================================================
st.title("RAG Chatbot")
st.caption("Conversational RAG with Reflection (Azure OpenAI + Qdrant)")


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
    # ---- USER MESSAGE
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # ---- REFLECTION (QUERY CONTEXTUALIZATION)
    rewritten_query = reflection.rewrite(
        st.session_state.messages,
        user_input
    )

    if rewritten_query != user_input:
        st.caption(f"🔎 Interpreted as: **{rewritten_query}**")

    # ---- ASSISTANT
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = rag.query(
                user_query=rewritten_query,
                top_k=top_k,
                mode=RAGMode.ONLINE
            )

            answer = result["answer"]
            documents = result.get("documents", [])

            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    # =====================================================
    # DOCUMENTS PANEL
    # =====================================================
    if documents:
        with st.expander("📄 Retrieved Documents", expanded=False):
            for i, doc in enumerate(documents, start=1):
                st.markdown(f"**Document {i}**")

                payload = doc.get("payload", {})
                text = payload.get("text", "")

                if text:
                    st.markdown(text)
                else:
                    st.caption("(No text in payload)")

                st.markdown("---")
