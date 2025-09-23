import json
from pathlib import Path
from typing import Any, Dict, List, Union


def load_any_json(path: Union[str, Path]) -> Union[List[Any], Dict[str, Any]]:
    """
    - JSON 배열([ ... ]) 또는 JSON 객체({ ... }) 파일: json.load 로 전체 파싱
    - 그 외: JSONL(라인별 JSON)로 간주하여 줄단위 파싱
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        # 선행 공백을 건너뛰고 첫 유의미 문자 확인
        head_chunk = f.read(4096)
        leading = head_chunk.lstrip()
        f.seek(0)

        if leading.startswith("[") or leading.startswith("{"):
            # 표준 JSON 파일(배열/객체)
            return json.load(f)
        else:
            # JSONL
            jsonl: List[Any] = []
            for i, line in enumerate(f, start=1):
                s = line.strip()
                if not s:
                    continue
                try:
                    jsonl.append(json.loads(s))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSONL at {path} line {i}: {e}") from e
            return jsonl
