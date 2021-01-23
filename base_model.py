# -*- coding: utf-8 -*-
"""
@author: longshuicui
@date  :   2020/9/18
@function: gRPC request
"""
import logging
import json
import grpc
import requests
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Model(object):
    def __init__(self, url=None, max_seq_length=None, tokenizer=None):
        if url is None:
            raise ValueError("service url is None")
        if max_seq_length is None:
            raise ValueError("should set max_seq_len")
        self.url = url
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

        logger.info(f"Loading {self.__class__.__name__}")

    def convert_features(self):
        raise NotImplementedError

    def gRPC_request(self, features):
        """
        gRPC request PB model
        :param features:dict {'signature_name':'',inputs:{'input_ids':''}}
        :return: result dict
        """
        channel = grpc.insecure_channel(target=self.url)
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel=channel)
        request = predict_pb2.PredictRequest()
        request.model_spec.name = features["signature_name"]
        request.model_spec.signature_name = features["signature_name"]
        for key, value in features["inputs"].items():
            request.inputs[key].CopyFrom(tf.contrib.util.make_tensor_proto(value))
        predict_future = stub.Predict.future(request, 10.0)  # timeout为10s
        result = predict_future.result()
        return result


class HttpModel:
    def __init__(self, url=None, max_seq_length=None, tokenizer=None):
        if url is None:
            raise ValueError("service url is None")
        if max_seq_length is None:
            raise ValueError("should set max_seq_len")
        self.url = "http://" + url + "/v1/models/serving_default"
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.predict_url = self.url + ":predict"

        status = self.get_model_status()
        if status["model_version_status"][0]["status"]["error_code"] != "OK":
            raise Exception("Domain classification model status is error! ")

        logger.info(f"{self.__class__.__name__} status: {status['model_version_status']}")

    def convert_features(self):
        raise NotImplementedError

    def get_model_status(self):
        results = requests.get(url=self.url)
        content = results.content.decode("utf8")
        return json.loads(content)

    def get_model_metadata(self):
        results = requests.get(url=self.url + "/metadata")
        content = results.content.decode("utf8")
        return content
