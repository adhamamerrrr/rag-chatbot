from pathlib import Path
from streamlit.testing.v1 import AppTest

def test_ui_initial_state():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
    assert not app.exception
    assert app.title[0].value == 'Engineering Document Assistant'
    assert app.chat_input[0].disabled
