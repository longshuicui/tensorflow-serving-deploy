# -*- coding: utf-8 -*-
"""
@author: longshuicui
@date  :   2021/1/23
@function:
"""
import os
import logging
import argparse
import tokenization
from flask import Flask, request, render_template
from extract_property_restful import ExtractShoesProperty

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="0.0.0.0", help="the flask host")
parser.add_argument("--port", type=str, default="8080", help="the flask port")
parser.add_argument("--url", type=str, default=None, help="host and port, 0.0.0.0:8500.")
parser.add_argument("--max_seq_length", type=int, default=128, help="the max sequence length.")
parser.add_argument("--vocab_file", type=str, default="./vocab.txt", help="the path of vocab to tokenizer")
# parser.add_argument("--do_lower_case", action="store_true", help="whether lower when tokenizer")
args = parser.parse_args()

logger.info("Loading tokenizer.")
tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=True)

logger.info("Loading Model.")
model = ExtractShoesProperty(url=args.url, max_seq_length=args.max_seq_length, tokenizer=tokenizer)


@app.route("/", methods=["GET"])
def home():
    return "OK"


@app.route("/extract", methods=["POST"])
def extract_entity():
    params = request.json
    text = params["text"]
    entities = model.predict(text)

    return {"status": "OK", "id": params["id"], "result": entities}


if __name__ == '__main__':
    app.run(host=args.host, port=args.port, debug=True)
