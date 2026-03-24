from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = REPO_ROOT / "test_outputs"
OUT_ROOT = TEST_ROOT / "aggregated_results"


@dataclass
class RunRecord:
    config: str
    method: str
    hp_config: str
    suite: str
    source_run_dir: str
    pretraining_src: str = ""
    probe_linear_src: str = ""
    probe_nonlinear_src: str = ""
    hp_config_inferred: bool = False
    notes: str = ""


def _safe_symlink(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(src.resolve(), target_is_directory=True)


def _load_metadata(path: Path) -> dict:
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return {}


def _write_metadata(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _merge_record(record: RunRecord) -> None:
    run_root = OUT_ROOT / record.config / record.method / record.hp_config
    run_root.mkdir(parents=True, exist_ok=True)

    pre_dst = run_root / "pretraining"
    lin_dst = run_root / "probe_linear"
    nonlin_dst = run_root / "probe_nonlinear"

    if record.pretraining_src:
        _safe_symlink(Path(record.pretraining_src), pre_dst)
    if record.probe_linear_src:
        _safe_symlink(Path(record.probe_linear_src), lin_dst)
    if record.probe_nonlinear_src:
        _safe_symlink(Path(record.probe_nonlinear_src), nonlin_dst)

    meta_path = run_root / "metadata.json"
    meta = _load_metadata(meta_path)
    sources = meta.get("sources", [])
    source_payload = asdict(record)
    if source_payload not in sources:
        sources.append(source_payload)

    meta.update(
        {
            "config": record.config,
            "method": record.method,
            "hp_config": record.hp_config,
            "stages_present": {
                "pretraining": pre_dst.exists() or pre_dst.is_symlink(),
                "probe_linear": lin_dst.exists() or lin_dst.is_symlink(),
                "probe_nonlinear": nonlin_dst.exists() or nonlin_dst.is_symlink(),
            },
            "sources": sources,
        }
    )
    _write_metadata(meta_path, meta)


def _make_record(
    *,
    config: str,
    method: str,
    seed_dir: Path,
    suite: str,
    hp_config: str,
    inferred: bool = False,
    notes: str = "",
) -> RunRecord:
    pre = seed_dir / "pretrain"
    probe = seed_dir / "probe"
    probe_linear = seed_dir / "probe_linear"
    probe_nonlinear = seed_dir / "probe_nonlinear"

    return RunRecord(
        config=config,
        method=method,
        hp_config=hp_config,
        suite=suite,
        source_run_dir=str(seed_dir.resolve()),
        pretraining_src=str(pre.resolve()) if pre.exists() else "",
        probe_linear_src=str((probe_linear if probe_linear.exists() else probe).resolve()) if (probe_linear.exists() or probe.exists()) else "",
        probe_nonlinear_src=str(probe_nonlinear.resolve()) if probe_nonlinear.exists() else "",
        hp_config_inferred=inferred,
        notes=notes,
    )


def collect_records() -> List[RunRecord]:
    records: List[RunRecord] = []

    # 1. Legacy lr/wd search
    root = TEST_ROOT / "v3_runs_A_lrwd_search"
    if root.exists():
        for seed_dir in root.glob("A*/**/seed_*"):
            if not (seed_dir.is_dir() and seed_dir.parent.name.startswith("lr_")):
                continue
            config = seed_dir.parents[2].name
            method = seed_dir.parents[1].name
            records.append(
                _make_record(
                    config=config,
                    method=method,
                    seed_dir=seed_dir,
                    suite="lrwd_search",
                    hp_config=seed_dir.parent.name,
                )
            )

    # 2. Family tuning optimizer + method
    for sub in ["optimizer", "method"]:
        root = TEST_ROOT / "v3_runs_A_family_tuning" / sub
        if root.exists():
            for seed_dir in root.glob("A*/**/seed_*"):
                if not seed_dir.is_dir():
                    continue
                hp_name = seed_dir.parent.name
                if not hp_name:
                    continue
                config = seed_dir.parents[2].name
                method = seed_dir.parents[1].name
                records.append(
                    _make_record(
                        config=config,
                        method=method,
                        seed_dir=seed_dir,
                        suite=f"family_{sub}",
                        hp_config=hp_name,
                    )
                )

    # 3. Selected search
    root = TEST_ROOT / "v3_runs_selected_search"
    if root.exists():
        for seed_dir in root.glob("A*/**/seed_*"):
            if not seed_dir.is_dir():
                continue
            hp_name = seed_dir.parent.name
            if hp_name.endswith(".png"):
                continue
            config = seed_dir.parents[2].name
            method = seed_dir.parents[1].name
            records.append(
                _make_record(
                    config=config,
                    method=method,
                    seed_dir=seed_dir,
                    suite="selected_search",
                    hp_config=hp_name,
                )
            )

    # 4. Manual A8/A11 all-method runs with known defaults
    root = TEST_ROOT / "v3_runs_A8_A11_all_methods"
    manual_hp = "lr_1e-3__wd_1e-4__tau_0p07__noise_0p1__cfw_0p5__mr_0p5__ema_0p996__var_1p0__kms_1__manual_all_methods"
    if root.exists():
        for seed_dir in root.glob("A*/**/seed_*"):
            if not seed_dir.is_dir():
                continue
            config = seed_dir.parents[1].name
            method = seed_dir.parents[0].name
            records.append(
                _make_record(
                    config=config,
                    method=method,
                    seed_dir=seed_dir,
                    suite="manual_all_methods",
                    hp_config=manual_hp,
                    inferred=True,
                    notes="hp_config inferred from the original manual run command defaults",
                )
            )

    # 5. Standalone nonlinear probes only
    root = TEST_ROOT / "nonlinear_probes_selected"
    if root.exists():
        for seed_dir in root.glob("A*/**/seed_*"):
            if not seed_dir.is_dir():
                continue
            config = seed_dir.parent.parent.name
            method = seed_dir.parent.name
            probe_nonlinear = seed_dir / "probe_nonlinear"
            if not probe_nonlinear.exists():
                continue
            records.append(
                RunRecord(
                    config=config,
                    method=method,
                    hp_config="standalone_nonlinear_seed_101",
                    suite="standalone_nonlinear",
                    source_run_dir=str(seed_dir.resolve()),
                    pretraining_src="",
                    probe_linear_src="",
                    probe_nonlinear_src=str(probe_nonlinear.resolve()),
                    hp_config_inferred=True,
                    notes="standalone nonlinear probe without colocated pretraining/linear artifacts",
                )
            )

    # 6. Best-lrwd nonlinear selected runs
    root = TEST_ROOT / "best_lrwd_nonlinear_selected"
    if root.exists():
        for seed_dir in root.glob("A*/**/seed_*"):
            if not seed_dir.is_dir():
                continue
            config = seed_dir.parents[2].name
            method = seed_dir.parents[1].name
            hp_config = "best_lrwd_selected"
            records.append(
                _make_record(
                    config=config,
                    method=method,
                    seed_dir=seed_dir,
                    suite="best_lrwd_nonlinear_selected",
                    hp_config=hp_config,
                    inferred=True,
                    notes="best lr/wd rerun with colocated nonlinear probe",
                )
            )

    # 7. Expanded method-hyperparameter selected runs
    root = TEST_ROOT / "expanded_method_hparam_search_selected"
    if root.exists():
        for seed_dir in root.glob("A*/**/seed_*"):
            if not seed_dir.is_dir():
                continue
            config = seed_dir.parents[2].name
            method = seed_dir.parents[1].name
            hp_config = seed_dir.parent.name
            records.append(
                _make_record(
                    config=config,
                    method=method,
                    seed_dir=seed_dir,
                    suite="expanded_method_hparam_selected",
                    hp_config=hp_config,
                )
            )

    return records


def write_index(records: List[RunRecord]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_ROOT / "aggregated_results_index.csv"
    json_path = OUT_ROOT / "aggregated_results_index.json"

    fieldnames = [
        "config",
        "method",
        "hp_config",
        "suite",
        "source_run_dir",
        "pretraining_src",
        "probe_linear_src",
        "probe_nonlinear_src",
        "hp_config_inferred",
        "notes",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))

    with json_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, indent=2)


def main() -> None:
    records = collect_records()
    for record in records:
        _merge_record(record)
    write_index(records)
    print(f"Aggregated {len(records)} source records into: {OUT_ROOT}")


if __name__ == "__main__":
    main()
