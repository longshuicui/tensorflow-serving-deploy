# -*- coding: utf-8 -*-
"""
@author: longshuicui
@date  :   2020/9/18
@function: extract goods property
"""
import os
import logging
import numpy as np
import tokenization, base_model

logger = logging.getLogger(__name__)


class ExtractShoesProperty(base_model.Model):
    """extract goods property"""

    @property
    def property_dict(self):
        return {0: 'O', 1: 'closure_type', 2: 'color', 3: 'features', 4: 'gender', 5: 'insole_material',
                6: 'lining_material', 7: 'material', 8: 'occasion', 9: 'outsole_material', 10: 'pattern_type',
                11: 'season', 12: 'style', 13: 'toe_shape', 14: 'upper_material'}

    @staticmethod
    def token_decode(text):
        tokens = []
        for t in text.split():
            if t[:2] == "##":
                if len(tokens) == 0:
                    tokens.append(t[2:])
                else:
                    tokens[-1] += t[2:]
            else:
                tokens.append(t)
        return " ".join(tokens)

    def convert_features(self, sentence):
        sentence = tokenization.convert_to_unicode(sentence)
        tokens_a = self.tokenizer.tokenize(sentence)
        if len(tokens_a) > self.max_seq_length - 2:
            tokens_a = tokens_a[0:(self.max_seq_length - 2)]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length
        start_label_id = [0]
        end_label_id = [0]
        input_features = (input_ids, input_mask, segment_ids, start_label_id, end_label_id)
        return tokens, input_features

    def convert_outputs(self, doc_tokens, start_points, end_points):
        """
        :param doc_tokens:
        :param start_points:
        :param end_points:
        :return: list(tuple) [(ent1, ent_type1), (ent2, ent_type2)]
        """
        start_points = np.argmax(start_points, axis=1)
        end_points = np.argmax(end_points, axis=1)
        # parse
        doc_length = len(doc_tokens)
        doc_start_labels = [self.property_dict[idx] for idx in start_points[:doc_length]]
        doc_end_labels = [self.property_dict[idx] for idx in end_points[:doc_length]]
        # extract property entity
        entities = []
        entity = []
        flag = None
        for index in range(min(len(doc_tokens), len(doc_start_labels), len(doc_end_labels))):
            start_label = doc_start_labels[index]
            end_label = doc_end_labels[index]
            if start_label == "O" and end_label == "O" and len(entity) == 0:
                continue
            if start_label == end_label and start_label != "O":
                entities.append((self.token_decode(doc_tokens[index]), start_label))
                entity = []
            else:
                if len(entity) == 0 and start_label != "O" and end_label == "O":
                    entity.append(doc_tokens[index])
                    flag = start_label
                elif len(entity) != 0 and start_label != "O" and end_label == "O":
                    if start_label == flag:
                        entity = [doc_tokens[index]]
                        flag = start_label
                    else:
                        entity.append(doc_tokens[index])
                elif len(entity) != 0 and start_label == "O" and end_label == "O":
                    entity.append(doc_tokens[index])
                elif len(entity) != 0 and start_label == "O" and end_label != "O":
                    if end_label == flag:
                        entity.append(doc_tokens[index])
                        text = self.token_decode(" ".join(entity))
                        entities.append((text, end_label))
                    entity = []

        return entities

    def predict(self, uttrance):
        tokens, input_features = self.convert_features(uttrance)
        features = {
            "signature_name": "serving_default",
            "inputs": {
                "input_ids": [input_features[0]],
                "input_mask": [input_features[1]],
                "segment_ids": [input_features[2]],
                "start_label_ids": input_features[3],
                "end_label_ids": input_features[4]
            }
        }

        results = self.gRPC_request(features=features)
        start_points = results.outputs["start_points"].float_val
        end_points = results.outputs["end_points"].float_val
        start_points = np.asarray(start_points).reshape([-1, len(self.property_dict)])
        end_points = np.asarray(end_points).reshape([-1, len(self.property_dict)])

        entities = self.convert_outputs(tokens, start_points, end_points)

        if len(entities) == 0:
            return {}

        entity_set = {}
        for ent, ent_type in entities:
            entity_set[ent_type] = entity_set.get(ent_type, set())
            entity_set[ent_type].add(ent)

        return {"property": {key: list(entity_set[key]) for key in entity_set}}
