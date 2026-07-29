"""Stage folder path mapping.

각 stage 는 별도 timestamp 폴더에 뽑혔음 (`run_stage_split.sh` 를 stage 별로 개별 실행).
같은 stage 를 재실행하면 이 매핑만 새 폴더로 갱신하면 됨.
"""
from __future__ import annotations
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent  # bos_model_copy/resnet50_jbpark/

STAGE_PATHS: dict[int, Path] = {
    0: _REPO_ROOT / "stage_split_20260712_173806" / "stage_0",
    1: _REPO_ROOT / "stage_split_20260718_143633" / "stage_1",
    2: _REPO_ROOT / "stage_split_20260718_144746" / "stage_2",
    3: _REPO_ROOT / "stage_split_20260718_151812" / "stage_3",
    4: _REPO_ROOT / "stage_split_20260718_155405" / "stage_4",
}


def validate() -> None:
    """Every stage folder exists and has logs/tracy_ops_data.csv + noc traces."""
    missing = []
    for st, p in STAGE_PATHS.items():
        logs = p / "logs"
        if not logs.is_dir():
            missing.append(f"stage_{st}: {logs} 없음")
            continue
        if not (logs / "tracy_ops_data.csv").is_file():
            missing.append(f"stage_{st}: tracy_ops_data.csv 없음")
        if not any(logs.glob("noc_trace_ID*_merged.json")):
            missing.append(f"stage_{st}: noc_trace_*_merged.json 없음")
    if missing:
        raise FileNotFoundError("\n".join(missing))


if __name__ == "__main__":
    validate()
    print("[OK] all 5 stage paths valid")
    for st, p in STAGE_PATHS.items():
        n = len(list((p / "logs").glob("noc_trace_ID*_merged.json")))
        print(f"  stage_{st}: {n:3d} merged JSONs  {p}")
