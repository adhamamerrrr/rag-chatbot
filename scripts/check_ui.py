"""Actual local model integration via Streamlit AppTest; first run downloads models."""
import os
from pathlib import Path
from streamlit.testing.v1 import AppTest
root = Path(__file__).resolve().parents[1]
os.chdir(root)
app = AppTest.from_file(str(root/'app.py'), default_timeout=60).run()
app.button[0].click().run(timeout=60)
assert not app.exception and 'index' in app.session_state
app.chat_input[0].set_value('What does an AAAA record contain?').run(timeout=60)
assert not app.exception and any('dns.txt' in x.label for x in app.expander)
app.checkbox[0].uncheck().run()
app.chat_input[0].set_value('What does an AAAA record contain?').run(timeout=60)
assert not app.exception
assert 'IPv6' in app.session_state['messages'][-1]['text']
assert app.session_state['messages'][-1]['sources'][0]['source'] == 'dns.txt'
print('Real embedding retrieval and local generation UI checks passed')
