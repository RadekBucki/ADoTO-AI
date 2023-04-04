from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello() -> str:
    return "HelloWorld"


@app.route('/test1')
def test() -> str:
    return "Example output"

