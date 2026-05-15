import time
import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from llmservice import call_llm
from prompts import build_prompt
from query_rewriter import rewrite_query
from rag import retrieve_context
from safety import is_safe_input, sanitize_context


st.set_page_config(
    page_title="Enterprise RAG Intelligence",
    page_icon=None,
    layout="wide",
)


def format_history(messages: list[dict[str, str]], limit: int = 5) -> str:
    recent_messages = messages[-limit * 2 :]
    history = ""

    for message in recent_messages:
        role = "User" if message["role"] == "user" else "Assistant"
        history += f"\n{role}: {message['content']}\n"

    return history


def stream_answer(answer: str):
    for word in answer.split():
        yield word + " "
        time.sleep(0.015)


def run_rag_pipeline(question: str, domain: str | None, history: str):
    start = time.perf_counter()

    rewritten_query = rewrite_query(question, history)

    metadata_filter = {"doc_type": domain} if domain else None
    context, sources, _debug_results = retrieve_context(rewritten_query, metadata_filter)

    if not context:
        context = "No company knowledge found."
    else:
        context = sanitize_context(context)

    prompt = build_prompt(history, context, question)
    answer = call_llm(prompt, mode="answer")
    response_time = time.perf_counter() - start
    sources = [str(source) for source in sources if source]

    return {
        "answer": answer,
        "context": context,
        "sources": sources,
        "rewritten_query": rewritten_query,
        "response_time": response_time,
    }


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_run" not in st.session_state:
    st.session_state.last_run = None


with st.sidebar:
    st.title("Enterprise RAG")

    domain_choice = st.selectbox(
        "Domain",
        ["All", "policy_terms", "refund_terms", "security_guidelines"],
    )
    selected_domain = None if domain_choice == "All" else domain_choice

    show_context = st.toggle("Show retrieved context", value=False)
    show_rewritten_query = st.toggle("Show rewritten query", value=False)

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_run = None
        st.rerun()

    st.caption(f"Session: {st.session_state.session_id[:8]}")


st.title("Enterprise RAG Intelligence System")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input("Ask about company policy, refunds, or security")

if question:
    question = question.strip()
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    if not is_safe_input(question):
        answer = "Request blocked by policy."
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.last_run = {
            "answer": answer,
            "context": "",
            "sources": [],
            "rewritten_query": question,
            "response_time": 0.0,
        }

        with st.chat_message("assistant"):
            st.markdown(answer)
    else:
        history = format_history(st.session_state.messages[:-1])

        with st.spinner("Generating answer..."):
            result = run_rag_pipeline(question, selected_domain, history)

        st.session_state.last_run = result
        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"]}
        )

        with st.chat_message("assistant"):
            st.write_stream(stream_answer(result["answer"]))


if st.session_state.last_run:
    result = st.session_state.last_run

    metric_cols = st.columns(2)
    metric_cols[0].metric("Response time", f"{result['response_time']:.2f}s")
    metric_cols[1].metric("Sources", len(result["sources"]))

    if result["sources"]:
        st.subheader("Sources")
        st.write(", ".join(result["sources"]))

    if show_rewritten_query and result["rewritten_query"]:
        st.subheader("Rewritten query")
        st.code(result["rewritten_query"])

    if show_context and result["context"]:
        st.subheader("Retrieved context")
        st.code(result["context"])
