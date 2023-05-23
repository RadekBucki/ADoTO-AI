from flask import Flask, request, redirect, jsonify
from functions import request_image

app = Flask(__name__)


@app.route('/photo', methods=["GET"])
def get_image():
    args = request.args
    return request_image(
        args.get("width"),
        args.get("minx"),
        args.get("miny"),
        args.get("maxx"),
        args.get("maxy"),
        "a.png")


def jsonify_response(response):
    res = jsonify({key: value for key, value in list(response.items())[:-1]}), response["status_code"]
    return res
