import json
import numpy as np
import pytest
from portfolio_rag.core import Chunk, load_chunks, SemanticIndex, validate_question

class Embeddings:
    def embed_documents(self, texts):
        return [[1., 0.] if 'network' in t else [0., 1.] for t in texts]
    def embed_query(self, text):
        return [1., 0.]

def test_chunk_offsets(tmp_path):
    text = 'abcdefghijklmnopqrstuvwxyz'
    (tmp_path / 'a.txt').write_text(text)
    chunks = load_chunks(tmp_path, size=10, overlap=2)
    assert len(chunks) == 3
    assert all(c.text == text[c.start_char:c.start_char + len(c.text)] for c in chunks)

def test_empty_and_invalid_chunks(tmp_path):
    with pytest.raises(ValueError):
        load_chunks(tmp_path)
    with pytest.raises(ValueError):
        load_chunks(tmp_path, size=10, overlap=10)

@pytest.mark.parametrize('question', ['', ' ', 'a' * 1001, None])
def test_question_validation(question):
    with pytest.raises(ValueError):
        validate_question(question)

def test_retrieval_and_roundtrip(tmp_path):
    chunks = [Chunk('a', 'network.txt', 0, 'network'), Chunk('b', 'robot.txt', 0, 'robot')]
    index = SemanticIndex(chunks, Embeddings())
    hits = index.retrieve('network', min_score=.5)
    assert len(hits) == 1 and hits[0]['source'] == 'network.txt'
    assert np.isclose(hits[0]['score'], 1.)
    index.save(tmp_path)
    assert SemanticIndex.load(tmp_path, Embeddings()).retrieve('network') == index.retrieve('network')
    metadata = json.loads((tmp_path / 'documents.json').read_text())
    metadata['chunks'][0]['text'] = 'tampered'
    (tmp_path / 'documents.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        SemanticIndex.load(tmp_path, Embeddings())

def test_invalid_search():
    index = SemanticIndex([Chunk('a', 'a.txt', 0, 'network')], Embeddings())
    with pytest.raises(ValueError):
        index.retrieve('question', k=0)
    with pytest.raises(ValueError):
        index.retrieve('question', min_score=float('nan'))

def test_abstention_and_complete_chunk_budget():
    from portfolio_rag.core import LocalGenerator
    generator = LocalGenerator.__new__(LocalGenerator)
    class Tokenizer:
        def encode(self, text):
            return list(range(len(text)))
    class Chain:
        def invoke(self, variables):
            assert 'short evidence' in variables['context']
            assert 'too long' not in variables['context']
            return ' answer '
    generator.tokenizer, generator.chain = Tokenizer(), Chain()
    assert generator.answer('question', [])['generated'] is False
    short = {'id': 's', 'text': 'short evidence'}
    long = {'id': 'l', 'text': 'too long' * 100}
    result = generator.answer('question', [long, short])
    assert result['answer'] == 'answer' and result['sources'] == [short]
    assert generator.answer('question', [long])['generated'] is False
