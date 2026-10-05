"""Command-line index construction and question answering."""
import argparse
import json
from pathlib import Path
from .core import load_chunks, SemanticIndex, make_embeddings, LocalGenerator


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--docs', type=Path, default=Path('sample_docs'))
    p.add_argument('--index', type=Path, default=Path('.rag_index'))
    p.add_argument('--rebuild', action='store_true')
    p.add_argument('--question', required=True)
    p.add_argument('--top-k', type=int, default=3)
    p.add_argument('--min-score', type=float, default=.25)
    p.add_argument('--retrieve-only', action='store_true')
    a = p.parse_args()
    try:
        embeddings = make_embeddings()
        if a.rebuild or not (a.index/'documents.json').exists():
            index = SemanticIndex(load_chunks(a.docs), embeddings)
            index.save(a.index)
        else:
            index = SemanticIndex.load(a.index, embeddings)
        hits = index.retrieve(a.question, a.top_k, a.min_score)
        result = {'answer': None, 'sources': hits, 'generated': False} if a.retrieve_only else LocalGenerator().answer(a.question, hits)
        print(json.dumps(result, indent=2))
    except (OSError, ValueError) as exc:
        p.exit(1, f'RAG error: {exc}\n')

if __name__ == '__main__':
    main()
