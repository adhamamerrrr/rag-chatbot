"""Small authored retrieval fixture and actual generation smoke test, not an academic benchmark."""
from pathlib import Path
import json
from datetime import datetime, timezone
from portfolio_rag.core import load_chunks, SemanticIndex, make_embeddings, LocalGenerator, EMBED_MODEL, EMBED_REVISION, GEN_MODEL, GEN_REVISION, fingerprint

QUESTIONS = [
    ('Which protocol assigns an IP address lease?', 'dhcp.txt'),
    ('What does an AAAA record contain?', 'dns.txt'),
    ('Why should oversampling happen after splitting?', 'smote.txt'),
    ('What is a blackhole attack?', 'wsn.txt'),
    ('How does AMCL localize a robot?', 'robotics.txt'),
    ('Why normalize embeddings for inner product search?', 'retrieval.txt'),
]
root = Path(__file__).resolve().parents[1]
chunks = load_chunks(root / 'sample_docs')
index = SemanticIndex(chunks, make_embeddings())
rows = []
for question, expected in QUESTIONS:
    hits = index.retrieve(question)
    rows.append({'question': question, 'expected_source': expected, 'top_source': hits[0]['source'] if hits else None,
                 'top_score': hits[0]['score'] if hits else None, 'top1_matches': bool(hits and hits[0]['source'] == expected)})
generator = LocalGenerator()
question = 'What does an AAAA record contain?'
answer = generator.answer(question, index.retrieve(question))
result = {'run_utc': datetime.now(timezone.utc).isoformat(), 'fixture': 'six authored engineering notes; six authored queries; no train/test split',
          'corpus_sha256': fingerprint(chunks), 'embedding_model': EMBED_MODEL, 'embedding_revision': EMBED_REVISION,
          'generator_model': GEN_MODEL, 'generator_revision': GEN_REVISION, 'retrieval': rows,
          'top1_matches': sum(row['top1_matches'] for row in rows), 'query_count': len(rows),
          'generation_smoke': {'question': question, **answer}, 'warning': 'Convenience smoke fixture; not evidence of general retrieval or answer accuracy.'}
(root/'results'/'smoke.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
