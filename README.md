# Document Q&A (RAG)

Upload a document, ask a question, and get an answer taken from its text. The app finds the passages that best match your question and then either shows them or, if you add an API key, has Claude write an answer from them.

## Features
- Upload `.txt`, `.md`, and `.pdf` files
- Ask questions in plain language and see which file the answer came from
- Works with no API key: shows the best matching sentences from the best matching passage
- Optional written answers from Claude, based only on the retrieved passages
- If the Claude call fails, falls back to the no-key answer
- Shows the passages used, with match scores
- A sample store FAQ loads when nothing is uploaded
- Files that can't be read are reported and skipped instead of crashing the app
- 44 automated tests (none need the network or an API key)

## How It Works
1. **Load**: read the uploaded files into text (Windows line endings are normalized)
2. **Chunk**: split the text into passages, keeping paragraphs together and overlapping long ones
3. **Retrieve**: rank passages against the question using TF-IDF similarity (scikit-learn)
4. **Answer**: pick the best sentences, or send the top passages to Claude and ask it to answer from them only

## Tech Stack
- Python 3.11
- Streamlit (interface)
- scikit-learn (TF-IDF retrieval)
- pypdf (PDF text extraction)
- Anthropic SDK (optional written answers)
- pytest

## Getting Started

```bash
git clone https://github.com/CodeNova55/rag-document-qa.git
cd rag-document-qa
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501

## Optional: Claude Answers

Set these environment variables (see `.env.example`) before starting the app:

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | your Anthropic API key; enables written answers |
| `ANTHROPIC_MODEL` | model to use (defaults to `claude-sonnet-5-5`) |

Without a key the app still works and shows the best matching passage.

## Running Tests

```bash
python -m pytest -v
```

## Project Structure

## Limitations
- Retrieval matches on words, not meaning, so a question phrased very differently from the document may miss
- Scanned PDFs (images with no text) are not supported
- Documents are re-read each session; nothing is saved between runs
- The Claude path is covered by tests that use a fake client; it has not yet been run end to end against the live API from this repo

## Git Workflow
- One feature branch per piece of work
- Feature branch -> PR into `develop`
- `develop` -> PR into `main` (after approval)
- No direct pushes to `develop` or `main`

## Roadmap
- Semantic search with embeddings
- Remember uploaded documents between sessions
- OCR for scanned PDFs
- Deployment
