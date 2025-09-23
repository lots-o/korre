import os
import torch
import numpy as np
import easydict
from pathlib import Path
from itertools import permutations


import warnings

warnings.filterwarnings("ignore")

from gliner import GLiNER
from transformers import AutoTokenizer, AutoModel
from transformers import logging

from .utils import load_any_json


base_path = Path(__file__).resolve().parent


class KorRE:
    def __init__(self):
        self.args = easydict.EasyDict(
            {
                "bert_model": "lots-o/kre-bert",
                "ner_model": "lots-o/gliner-bi-ko-xlarge-v1",
                "entity_label": os.path.join(base_path, "entity_label.json"),
                "relid2label": os.path.join(base_path, "relid2label.json"),
                "mode": "ALLCC",
                "n_class": 97,
                "max_token_len": 512,
                "max_acc_threshold": 0.6,
            }
        )
        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        want_dtype = torch.float32
        self.ner_module = GLiNER.from_pretrained(self.args.ner_model, torch_dtype=want_dtype)
        self.ner_module.to(self.device, dtype=want_dtype)
        self.ner_module.eval()

        logging.set_verbosity_error()

        self.tokenizer = AutoTokenizer.from_pretrained(self.args.bert_model)
        self.trained_model = AutoModel.from_pretrained(self.args.bert_model, trust_remote_code=True)
        self.trained_model.to(self.device)
        self.trained_model.eval()

        # relation id to label
        self.relid2label = load_any_json(self.args.relid2label)

        # entity label
        self.entity_label = []
        for _, entities in load_any_json(self.args.entity_label).items():
            self.entity_label.extend(entities.values())

        # relation list
        self.relation_list = list(self.relid2label.keys())

        # Pre-encode entity labels for faster NER inference.
        self.entity_embeddings = self.ner_module.encode_labels(self.entity_label, batch_size=8)
        self.entity_embeddings = self.entity_embeddings.to(device=self.device, dtype=want_dtype)

    def __idx2relid(self, idx_list):
        """onehot label에서 1인 위치 인덱스 리스트를 relation id 리스트로 변환하는 함수.

        Example:
            relation_list = ['P17', 'P131', 'P530', ...] 일 때,
            __idx2relid([0, 2]) => ['P17', 'P530'] 을 반환.
        """
        label_out = []

        for idx in idx_list:
            label = self.relation_list[idx]
            label_out.append(label)

        return label_out

    def gliner_ner(self, sentence: str):
        """gliner의 ner 모듈을 이용하여 그대로 반환하는 함수."""
        return self.ner_module.predict_with_embeds(sentence, labels_embeddings=self.entity_embeddings, labels=self.entity_label, threshold=0.5)

    def ner(self, sentence: str):
        """주어진 문장에서 gliner의 ner 모듈을 이용하여 개체명 인식을 수행하고 각 개체의 인덱스 위치를 함께 반환하는 함수."""
        # gliner_ner는 [[{'text':..., 'label':..., 'start':..., 'end':...}]] 형태의 결과를 반환합니다.
        ner_results_for_sentence = self.gliner_ner(sentence)
        if not ner_results_for_sentence:
            return []

        entities = ner_results_for_sentence
        return [(entity["text"], entity["label"], [entity["start"], entity["end"]]) for entity in entities]

    def get_all_entity_pairs(self, sentence: str) -> list:
        """주어진 문장에서 개체명 인식을 통해 모든 가능한 [문장, subj_range, obj_range]의 리스트를 반환하는 함수.

        Example:
            sentence = '모토로라 레이저 M는 모토로라 모빌리티에서 제조/판매하는 안드로이드 스마트폰이다.'

        Return:
            [(('모토로라 레이저 M', 'ARTIFACT', [0, 10]), ('모토로라 모빌리티', 'ORGANIZATION', [12, 21])),
             (('모토로라 레이저 M', 'ARTIFACT', [0, 10]), ('안드로이드', 'TERM', [32, 37])),
             (('모토로라 레이저 M', 'ARTIFACT', [0, 10]), ('스마트폰', 'TERM', [38, 42])),
             (('모토로라 모빌리티', 'ORGANIZATION', [12, 21]), ('모토로라 레이저 M', 'ARTIFACT', [0, 10])),
             (('모토로라 모빌리티', 'ORGANIZATION', [12, 21]), ('안드로이드', 'TERM', [32, 37])),
             (('모토로라 모빌리티', 'ORGANIZATION', [12, 21]), ('스마트폰', 'TERM', [38, 42])),
             (('안드로이드', 'TERM', [32, 37]), ('모토로라 레이저 M', 'ARTIFACT', [0, 10])),
             (('안드로이드', 'TERM', [32, 37]), ('모토로라 모빌리티', 'ORGANIZATION', [12, 21])),
             (('안드로이드', 'TERM', [32, 37]), ('스마트폰', 'TERM', [38, 42])),
             (('스마트폰', 'TERM', [38, 42]), ('모토로라 레이저 M', 'ARTIFACT', [0, 10])),
             (('스마트폰', 'TERM', [38, 42]), ('모토로라 모빌리티', 'ORGANIZATION', [12, 21])),
             (('스마트폰', 'TERM', [38, 42]), ('안드로이드', 'TERM', [32, 37]))]
        """
        # 너무 긴 문장의 경우 500자 이내로 자름
        if len(sentence) >= 500:
            sentence = sentence[:499]

        ent_list = self.ner(sentence)

        pairs = list(permutations(ent_list, 2))

        return pairs

    def get_all_inputs(self, sentence: str) -> list:
        """주어진 문장에서 관계 추출 모델에 통과시킬 수 있는 모든 input의 리스트를 반환하는 함수.

        Example:
            sentence = '모토로라 레이저 M는 모토로라 모빌리티에서 제조/판매하는 안드로이드 스마트폰이다.'

        Return:
            [['모토로라 레이저 M는 모토로라 모빌리티에서 제조/판매하는 안드로이드 스마트폰이다.', [0, 10], [12, 21]],
            ['모토로라 레이저 M는 모토로라 모빌리티에서 제조/판매하는 안드로이드 스마트폰이다.', [0, 10], [32, 37]],
            ..., ]
        """
        pairs = self.get_all_entity_pairs(sentence)
        return [[sentence, ent_subj[2], ent_obj[2]] for ent_subj, ent_obj in pairs]

    def entity_markers_added(self, sentence: str, subj_range: list, obj_range: list) -> str:
        """문장과 관계를 구하고자 하는 두 개체의 인덱스 범위가 주어졌을 때 entity marker token을 추가하여 반환하는 함수.

        Example:
            sentence = '모토로라 레이저 M는 모토로라 모빌리티에서 제조/판매하는 안드로이드 스마트폰이다.'
            subj_range = [0, 10]   # sentence[subj_range[0]: subj_range[1]] => '모토로라 레이저 M'
            obj_range = [12, 21]   # sentence[obj_range[0]: obj_range[1]] => '모토로라 모빌리티'

        Return:
            '[E1] 모토로라 레이저 M [/E1] 는  [E2] 모토로라 모빌리티 [/E2] 에서 제조/판매하는 안드로이드 스마트폰이다.'
        """
        e1_s, e1_e, e2_s, e2_e = self.trained_model.config.marker_tokens
        result_sent = ""

        for i, char in enumerate(sentence):
            if i == subj_range[0]:
                result_sent += f" {e1_s} "
            elif i == subj_range[1]:
                result_sent += f" {e1_e} "
            if i == obj_range[0]:
                result_sent += f" {e2_s} "
            elif i == obj_range[1]:
                result_sent += f" {e2_e} "
            result_sent += sentence[i]
        if subj_range[1] == len(sentence):
            result_sent += f" {e1_e}"
        elif obj_range[1] == len(sentence):
            result_sent += f" {e2_e}"

        return result_sent.strip()

    def infer(
        self,
        sentence: str,
        subj_range=None,
        obj_range=None,
        entity_markers_included=False,
    ):
        """입력받은 문장에 대해 관계 추출 태스크를 수행하는 함수."""
        e1_s, e1_e, e2_s, e2_e = self.trained_model.config.marker_ids

        # entity marker token이 포함된 경우
        if entity_markers_included:
            # subj, obj name 구하기
            tmp_input_ids = self.tokenizer(sentence)["input_ids"]

            if tmp_input_ids.count(e1_s) != 1 or tmp_input_ids.count(e1_e) != 1 or tmp_input_ids.count(e2_s) != 1 or tmp_input_ids.count(e2_e) != 1:
                raise Exception("Incorrect number of entity marker tokens.")

            subj_start_id, subj_end_id = tmp_input_ids.index(e1_s), tmp_input_ids.index(e1_e)
            obj_start_id, obj_end_id = tmp_input_ids.index(e2_s), tmp_input_ids.index(e2_e)

            subj_name = self.tokenizer.decode(tmp_input_ids[subj_start_id + 1 : subj_end_id])
            obj_name = self.tokenizer.decode(tmp_input_ids[obj_start_id + 1 : obj_end_id])

            encoding = self.tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                max_length=self.args.max_token_len,
                return_token_type_ids=False,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].to(self.device)
            mask = encoding["attention_mask"].to(self.device)

            prediction = self.trained_model(input_ids, mask)["probs"]

            predictions = [prediction.flatten()]
            predictions = torch.stack(predictions).detach().cpu()

            y_pred = predictions.numpy()
            upper, lower = 1, 0
            y_pred = np.where(y_pred > self.args.max_acc_threshold, upper, lower)

            preds_list = []

            for i in range(len(y_pred)):
                class_pred = self.__idx2relid(np.where(y_pred[i] == 1)[0])
                preds_list.append(class_pred)

            preds_list = preds_list[0]

            pred_rel_list = [self.relid2label[pred] for pred in preds_list]

            return [(subj_name, obj_name, pred_rel) for pred_rel in pred_rel_list]

        # entity_markers_included=False인 경우
        else:
            # entity marker가 문장에 포함된 경우
            tmp_input_ids = self.tokenizer(sentence)["input_ids"]
            if tmp_input_ids.count(e1_s) >= 1 or tmp_input_ids.count(e1_e) >= 1 or tmp_input_ids.count(e2_s) >= 1 or tmp_input_ids.count(e2_e) >= 1:
                raise Exception("Entity marker tokens already exist in the input sentence. Try 'entity_markers_included=True'.")

            # subj range와 obj range가 주어진 경우
            if subj_range is not None and obj_range is not None:
                # add entity markers
                converted_sent = self.entity_markers_added(sentence, subj_range, obj_range)

                encoding = self.tokenizer.encode_plus(
                    converted_sent,
                    add_special_tokens=True,
                    max_length=self.args.max_token_len,
                    return_token_type_ids=False,
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )

                input_ids = encoding["input_ids"].to(self.device)
                mask = encoding["attention_mask"].to(self.device)

                prediction = self.trained_model(input_ids, mask)["probs"]

                predictions = [prediction.flatten()]
                predictions = torch.stack(predictions).detach().cpu()

                y_pred = predictions.numpy()
                upper, lower = 1, 0
                y_pred = np.where(y_pred > self.args.max_acc_threshold, upper, lower)

                preds_list = []

                for i in range(len(y_pred)):
                    class_pred = self.__idx2relid(np.where(y_pred[i] == 1)[0])
                    preds_list.append(class_pred)

                preds_list = preds_list[0]

                pred_rel_list = [self.relid2label[pred] for pred in preds_list]

                return [
                    (
                        sentence[subj_range[0] : subj_range[1]],
                        sentence[obj_range[0] : obj_range[1]],
                        pred_rel,
                    )
                    for pred_rel in pred_rel_list
                ]

            # 문장만 주어진 경우: 모든 경우에 대해 inference 수행
            else:
                input_list = self.get_all_inputs(sentence)

                converted_sent_list = [self.entity_markers_added(*input_list[i]) for i in range(len(input_list))]

                encoding_list = []

                for i, converted_sent in enumerate(converted_sent_list):
                    tmp_encoding = self.tokenizer.encode_plus(
                        converted_sent,
                        add_special_tokens=True,
                        max_length=self.args.max_token_len,
                        return_token_type_ids=False,
                        padding="max_length",
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors="pt",
                    )
                    encoding_list.append(tmp_encoding)

                predictions = []

                for i, item in enumerate(encoding_list):
                    prediction = self.trained_model(
                        item["input_ids"].to(self.device),
                        item["attention_mask"].to(self.device),
                    )["probs"]

                    predictions.append(prediction.flatten())

                if predictions:
                    predictions = torch.stack(predictions).detach().cpu()

                    y_pred = predictions.numpy()
                    upper, lower = 1, 0
                    y_pred = np.where(y_pred > self.args.max_acc_threshold, upper, lower)

                    preds_list = []
                    for i in range(len(y_pred)):
                        class_pred = self.__idx2relid(np.where(y_pred[i] == 1)[0])
                        preds_list.append(class_pred)

                    result_list = []
                    for i, input_i in enumerate(input_list):
                        tmp_subj_range, tmp_obj_range = input_i[1], input_i[2]
                        result_list.append(
                            (
                                sentence[tmp_subj_range[0] : tmp_subj_range[1]],
                                sentence[tmp_obj_range[0] : tmp_obj_range[1]],
                                preds_list[i],
                            )
                        )

                    final_list = []
                    for tmp_subj, tmp_obj, tmp_list in result_list:
                        for i in range(len(tmp_list)):
                            final_list.append((tmp_subj, tmp_obj, tmp_list[i]))

                    return [(item[0], item[1], self.relid2label[item[2]]) for item in final_list]

                else:
                    return []
