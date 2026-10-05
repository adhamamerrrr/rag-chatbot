"""Local Streamlit document assistant; models load only on request."""
from pathlib import Path
import streamlit as st
from portfolio_rag.core import load_chunks, fingerprint, make_embeddings, SemanticIndex, LocalGenerator

st.set_page_config(page_title='Engineering Document Assistant', page_icon='📚')
st.title('Engineering Document Assistant')
st.caption('Semantic document search with visible evidence')

@st.cache_resource
def build_index(chunks_key, _chunks):
    return SemanticIndex(_chunks, make_embeddings())

@st.cache_resource
def generator():
    return LocalGenerator()

with st.sidebar:
    folder = st.text_input('Document folder', 'sample_docs')
    retrieval_only = st.checkbox('Search without generating an answer', value=True)
    st.caption('First use downloads public models. Documents stay on this server.')
    if st.button('Load documents'):
        try:
            chunks = load_chunks(Path(folder))
            with st.spinner('Loading semantic search…'):
                st.session_state.index = build_index(fingerprint(chunks), chunks)
            st.session_state.messages = []
            st.success(f'Loaded {len(chunks)} excerpts')
        except Exception as exc:
            st.error(f'Could not load documents: {exc}')
    if st.button('Clear conversation'):
        st.session_state.messages = []

for message in st.session_state.get('messages', []):
    with st.chat_message(message['role']):
        st.write(message['text'])
        for hit in message.get('sources', []):
            with st.expander(f"{hit['source']} · similarity {hit['score']:.3f}"):
                st.write(hit['text'])

question = st.chat_input('Ask about the loaded documents', disabled='index' not in st.session_state)
if question:
    st.session_state.setdefault('messages', []).append({'role': 'user', 'text': question})
    with st.chat_message('user'):
        st.write(question)
    try:
        with st.spinner('Searching documents…'):
            hits = st.session_state.index.retrieve(question)
            if retrieval_only:
                result = {'answer': 'Matching excerpts are shown below.' if hits else 'No matching excerpts found.', 'sources': hits}
            else:
                result = generator().answer(question, hits)
        message = {'role': 'assistant', 'text': result['answer'], 'sources': result['sources']}
        st.session_state.messages.append(message)
        with st.chat_message('assistant'):
            st.write(message['text'])
            for hit in message['sources']:
                with st.expander(f"{hit['source']} · similarity {hit['score']:.3f}"):
                    st.write(hit['text'])
    except Exception as exc:
        st.error(f'Could not answer: {exc}')
