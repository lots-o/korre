import os
import hashlib
from typing import List
from itertools import permutations
from importlib import resources


import warnings

warnings.filterwarnings("ignore")

import torch
import easydict
from gliner import GLiNER
from transformers import AutoTokenizer, AutoModel
from transformers import logging

from .utils import load_any_json


class KorRE:
    def __init__(self, cache_dir: str = None):
        # 패키지 데이터 파일 경로 설정 (importlib.resources 사용)
        pkg_files = resources.files("korre")

        if cache_dir is None:
            cache_dir = pkg_files / "hf_cache"
        self.cache_dir = str(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.args = easydict.EasyDict(
            {
                "bert_model": "lots-o/kre-bert",
                "ner_model": "lots-o/gliner-bi-ko-xlarge-v1",
                "entity_label": pkg_files / "entity_label.json",
                "relid2label": pkg_files / "relid2label.json",
                "mode": "ALLCC",
                "n_class": 97,
                "max_token_len": 512,
                "max_acc_threshold": 0.6,
                "re_batch_size": 64,  # RE 추론 배치 크기
                "ner_threshold": 0.5,  # GLiNER ner 임계 값
            }
        )
        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_cuda = torch.cuda.is_available() and self.device.type == "cuda"

        want_dtype = torch.float32
        self.ner_module = GLiNER.from_pretrained(self.args.ner_model, torch_dtype=want_dtype, cache_dir=self.cache_dir)
        self.ner_module.to(self.device, dtype=want_dtype, non_blocking=self.use_cuda)
        self.ner_module.eval()

        logging.set_verbosity_error()

        self.tokenizer = AutoTokenizer.from_pretrained(self.args.bert_model, cache_dir=self.cache_dir)
        self.trained_model = AutoModel.from_pretrained(self.args.bert_model, trust_remote_code=True, cache_dir=self.cache_dir)
        self.trained_model.to(self.device, non_blocking=self.use_cuda)
        self.trained_model.eval()

        # relation id to label
        self.relid2label = load_any_json(self.args.relid2label)

        # entity label
        self.entity_label = []
        for _, entities in load_any_json(self.args.entity_label).items():
            self.entity_label.extend(entities.values())

        # relation list
        self.relation_list = list(self.relid2label.keys())

        # Pre-encode entity labels for faster NER inference with on-disk cache
        # 캐시 디렉토리 준비
        self.cache_dir = pkg_files / "cache"
        self.cache_dir.mkdir(exist_ok=True)

        # 캐시 키 & 경로
        cache_key = self._make_label_cache_key(self.entity_label, self.args.ner_model, want_dtype)
        cache_path = self.cache_dir / f"gliner_label_embeds_{cache_key}.pt"

        if cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu")
            if isinstance(cached, torch.Tensor):
                print(f"[KorRE] Loading cached GLiNER label embeddings from: {cache_path}")
                self.entity_embeddings = cached.to(device=self.device, dtype=want_dtype, non_blocking=self.use_cuda)
            else:
                # 텐서가 아니면 재생성
                print(f"[KorRE] Invalid cache file at {cache_path}. Regenerating...")
                self.entity_embeddings = self.ner_module.encode_labels(self.entity_label, batch_size=8)
                self.entity_embeddings = self.entity_embeddings.to(device=self.device, dtype=want_dtype, non_blocking=self.use_cuda)
                torch.save(self.entity_embeddings.to("cpu"), cache_path)
                print(f"[KorRE] Saved regenerated embeddings to: {cache_path}")
        else:
            # 최초 첫 저장
            print(f"[KorRE] GLiNER label embeddings cache not found. Generating...")
            self.entity_embeddings = self.ner_module.encode_labels(self.entity_label, batch_size=8)
            self.entity_embeddings = self.entity_embeddings.to(device=self.device, dtype=want_dtype, non_blocking=self.use_cuda)
            torch.save(self.entity_embeddings.to("cpu"), cache_path)
            print(f"[KorRE] Saved new embeddings to: {cache_path}")

    def _make_label_cache_key(self, labels: List[str], ner_model: str, dtype: torch.dtype) -> str:
        """labels + ner_model + dtype을 묶어서 캐시 키 생성"""
        h = hashlib.sha256()
        h.update(ner_model.encode("utf-8"))
        h.update(str(dtype).encode("utf-8"))
        for lab in sorted(labels):
            h.update(lab.encode("utf-8"))
        return h.hexdigest()[:16]

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
        return self.ner_module.predict_with_embeds(
            sentence, labels_embeddings=self.entity_embeddings, labels=self.entity_label, threshold=float(self.args.ner_threshold)
        )

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
        """입력받은 문장에 대해 관계 추출 태스크를 수행하는 함수.
        - 불필요한 그래프 생성을 막기 위해 inference_mode 사용
        - 다수의 (문장, 엔티티쌍) 케이스는 배치 인코딩/추론으로 GPU 메모리 사용을 최소화
        """
        e1_s, e1_e, e2_s, e2_e = self.trained_model.config.marker_ids
        threshold = float(self.args.max_acc_threshold)

        # ---------------------------
        # Helper: 배치 추론
        # ---------------------------
        def _predict_probs_batch(text_list: List[str]):
            """문장 리스트를 한 번에 토크나이즈/전달하여 [B, num_labels] 확률 텐서(CPU)를 반환"""
            if not text_list:
                return torch.empty(0, self.args.n_class)

            enc = self.tokenizer(
                text_list,
                add_special_tokens=True,
                max_length=self.args.max_token_len,
                padding=True,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            with torch.inference_mode():
                input_ids = enc["input_ids"].to(self.device, non_blocking=self.use_cuda)
                mask = enc["attention_mask"].to(self.device, non_blocking=self.use_cuda)
                out = self.trained_model(input_ids, mask)

                # 모델은 {"probs": [B, n_class]}를 반환 ( 확률)
                probs = out["probs"]

                # 바로 임계값 적용 후 CPU로 이동 (GPU 메모리 즉시 릴리즈를 위해)
                binarized = (probs >= threshold).to(torch.uint8).cpu()

            # 임시 텐서 정리
            del enc, input_ids, mask, out, probs
            if torch.cuda.is_available() and self.device.type == "cuda":
                torch.cuda.empty_cache()
            return binarized

        # ---------------------------
        # 1) 엔티티 마커가 포함된 경우 = (entity_markers_included=True)
        # ---------------------------
        if entity_markers_included:
            # subj, obj name 구하기 (마커 사이 span 디코딩)
            tmp_input_ids = self.tokenizer(sentence)["input_ids"]
            if tmp_input_ids.count(e1_s) != 1 or tmp_input_ids.count(e1_e) != 1 or tmp_input_ids.count(e2_s) != 1 or tmp_input_ids.count(e2_e) != 1:
                raise Exception("Incorrect number of entity marker tokens.")

            subj_start_id, subj_end_id = tmp_input_ids.index(e1_s), tmp_input_ids.index(e1_e)
            obj_start_id, obj_end_id = tmp_input_ids.index(e2_s), tmp_input_ids.index(e2_e)
            subj_name = self.tokenizer.decode(tmp_input_ids[subj_start_id + 1 : subj_end_id])
            obj_name = self.tokenizer.decode(tmp_input_ids[obj_start_id + 1 : obj_end_id])

            # 단일 문장 배치 추론
            preds_bin = _predict_probs_batch([sentence])  # [1, n_class] (CPU, uint8)
            active_idx = torch.nonzero(preds_bin[0]).flatten().tolist()
            pred_rel_ids = self.__idx2relid(active_idx)
            return [(subj_name, obj_name, self.relid2label[p]) for p in pred_rel_ids]

        # ---------------------------
        # 2) 엔티티 마커가 포함되지 않은 경우 (entity_markers_included=False)
        # ---------------------------
        # NOTE: 엔티티 마커가 이미 포함된 경우 예외
        tmp_input_ids = self.tokenizer(sentence)["input_ids"]
        if tmp_input_ids.count(e1_s) >= 1 or tmp_input_ids.count(e1_e) >= 1 or tmp_input_ids.count(e2_s) >= 1 or tmp_input_ids.count(e2_e) >= 1:
            raise Exception("Entity marker tokens already exist in the input sentence. Try 'entity_markers_included=True'.")

        # 2-a) (subj_range, obj_range) 가 주어진 경우: 단일 케이스만 처리
        if subj_range is not None and obj_range is not None:
            converted_sent = self.entity_markers_added(sentence, subj_range, obj_range)
            preds_bin = _predict_probs_batch([converted_sent])  # [1, n_class]
            active_idx = torch.nonzero(preds_bin[0]).flatten().tolist()
            pred_rel_ids = self.__idx2relid(active_idx)
            pred_rel_list = [self.relid2label[p] for p in pred_rel_ids]
            subj_text = sentence[subj_range[0] : subj_range[1]]
            obj_text = sentence[obj_range[0] : obj_range[1]]
            return [(subj_text, obj_text, rel) for rel in pred_rel_list]

        # 2-b) 문장만 주어진 경우: 가능한 모든 엔티티쌍에 대해 배치 처리
        input_list = self.get_all_inputs(sentence)  # [[sent, e1_range, e2_range], ...]
        if not input_list:
            return []

        # 엔티티 마커 삽입을 모두 미리 수행
        converted_sent_list = [self.entity_markers_added(*inp) for inp in input_list]

        # 배치 추론 (긴 목록은 chunk로 나눠 처리)
        BATCH = int(self.args.re_batch_size)
        all_bins = []
        for i in range(0, len(converted_sent_list), BATCH):
            chunk = converted_sent_list[i : i + BATCH]
            preds_bin = _predict_probs_batch(chunk)  # [len(chunk), n_class]
            all_bins.append(preds_bin)
        if not all_bins:
            return []
        preds_bin_all = torch.cat(all_bins, dim=0)  # [N, n_class] (CPU)

        # 각 케이스별로 활성 라벨을 수집
        result = []
        for row_i in range(preds_bin_all.size(0)):
            idxs = torch.nonzero(preds_bin_all[row_i]).flatten().tolist()
            rel_ids = self.__idx2relid(idxs)
            subj_range_i, obj_range_i = input_list[row_i][1], input_list[row_i][2]
            subj_text = sentence[subj_range_i[0] : subj_range_i[1]]
            obj_text = sentence[obj_range_i[0] : obj_range_i[1]]
            for rid in rel_ids:
                result.append((subj_text, obj_text, self.relid2label[rid]))

        # 중복 제거
        if result:
            result = list(dict.fromkeys(result))
        return result
