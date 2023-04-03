from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello():
    return "HelloWorld"


@app.route('/test1')
def hello():
    return "Example output"

