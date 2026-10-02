import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402

from docqa.answerer import default_client  # noqa: E402
from docqa.pipeline import ask, build_index  # noqa: E402

SAMPLE_DOC = ROOT / "sample_docs" / "duka_faq.txt"

st.set_page_config(page_title="Document Q&A", page_icon="📄")
st.title("Document Q&A")
st.caption("Upload a document, ask a question, and get an answer from its text.")

client = default_client()


@st.cache_resource(show_spinner="Reading documents...")
def get_index(documents):
    return build_index(list(documents))


with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader(
        "Upload .txt, .md, or .pdf files",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )
    if client:
        st.success("Claude answers are on.")
    else:
        st.info(
            "No API key set, so the app shows the best matching passage. "
            "Set ANTHROPIC_API_KEY to get written answers."
        )

documents = [(f.name, f.getvalue()) for f in uploads]
if not documents:
    documents = [(SAMPLE_DOC.name, SAMPLE_DOC.read_bytes())]
    st.sidebar.caption("Showing the sample store FAQ. Upload your own files to replace it.")

retriever, problems = get_index(tuple(documents))
for problem in problems:
    st.warning(problem)

question = st.text_input("Ask a question about your documents")

if question.strip():
    with st.spinner("Thinking..."):
        answer, results = ask(retriever, question, client=client)

    st.subheader("Answer")
    st.write(answer.text)
    if answer.note:
        st.caption(answer.note)
    if answer.sources:
        st.caption("Sources: " + ", ".join(answer.sources))

    if results:
        with st.expander("Passages used"):
            for result in results:
                st.markdown(f"**{result.chunk.source}** (match {result.score:.2f})")
                st.write(result.chunk.text)
