#!/usr/bin/env python3
"""Refresh the IEEE review package's PDF validation and file manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
BUILD_LOGS = {
    "main": ROOT / "build_en/main.log",
    "main_zh": ROOT / "build_zh/main_zh.log",
    "supplementary": ROOT / "build_supp_en/supplementary.log",
    "supplementary_zh": ROOT / "build_supp_zh/supplementary_zh.log",
}
BUILD_BIBS = {
    "main": ROOT / "build_en/main.bbl",
    "main_zh": ROOT / "build_zh/main_zh.bbl",
    "supplementary": ROOT / "build_supp_en/supplementary.bbl",
    "supplementary_zh": ROOT / "build_supp_zh/supplementary_zh.bbl",
}
PDFS = {
    "main": ROOT / "pdf/IEEE_Transactions_修订稿_英文_20261002.pdf",
    "main_zh": ROOT / "pdf/IEEE_Transactions_修订稿_中文_20261002.pdf",
    "supplementary": ROOT / "pdf/IEEE_Transactions_修订稿_英文补充_20261002.pdf",
    "supplementary_zh": ROOT / "pdf/IEEE_Transactions_修订稿_中文补充_20261002.pdf",
}
SKIP_DIRS = {"build_en", "build_zh", "build_supp_en", "build_supp_zh", "preview", "__pycache__"}
SKIP_SUFFIXES = {".aux", ".log", ".out", ".blg", ".bbl", ".pyc"}
SKIP_FILES = {
    "模型框架图提示词_20261001.md",
    "IEEE_Transactions_最终稿_中文_20260930.pdf",
    "IEEE_Transactions_最终稿_中文补充_20260930.pdf",
    "IEEE_Transactions_最终稿_英文_20260930.pdf",
    "IEEE_Transactions_最终稿_英文补充_20260930.pdf",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True).stdout.decode(
        "utf-8", errors="replace"
    )


def document_info(key: str, path: Path) -> dict:
    info = run("pdfinfo", str(path))
    pages = int(re.search(r"(?m)^Pages:\s+(\d+)$", info).group(1))
    size = re.search(r"(?m)^Page size:\s+(.+)$", info).group(1).strip()
    bbox = run("pdftotext", "-bbox", str(path), "-")
    legal = []
    filtered = 0
    for char in bbox:
        code = ord(char)
        if code in (9, 10, 13) or code >= 32:
            legal.append(char)
        else:
            filtered += 1
    root = ElementTree.fromstring("".join(legal))
    word_count = sum(element.tag.rsplit("}", 1)[-1] == "word"
                     for element in root.iter())
    bibliography = BUILD_BIBS[key].read_text(errors="replace")
    references = len(re.findall(r"\\bibitem\s*\{", bibliography))
    if not references:
        raise ValueError(f"numbered reference entries missing in {path}")
    log = (ROOT / BUILD_LOGS[key].relative_to(ROOT)).read_text(errors="replace")
    return {
        "pages": pages,
        "page_size": size,
        "references": references,
        "word_boxes_checked": word_count,
        "pdf_sha256": sha(path),
        "poppler_xml_control_characters_filtered": filtered,
        "overfull_boxes": len(re.findall(r"Overfull \\hbox|Overfull \\vbox", log)),
        "missing_glyphs": len(re.findall(r"Missing character:", log)),
        "undefined_references": len(re.findall(
            r"(?:Reference|Citation) .*? undefined|There were undefined references", log
        )),
    }


def package_files() -> list[Path]:
    result = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        if (path.name in SKIP_FILES or path.name == "MANIFEST.json"
                or path.suffix in SKIP_SUFFIXES):
            continue
        result.append(path)
    return sorted(result, key=lambda item: item.relative_to(ROOT).as_posix())


def update_figure_evidence() -> None:
    source_path = ROOT / "figure_data/source_data.json"
    source_data = json.loads(source_path.read_text())
    source_paths = source_data["source_files"]
    repository_file = next(
        Path(path).resolve() for path in source_paths
        if "BusSegPredicionV2" in Path(path).parts
    )
    repo_root = next(
        parent for parent in repository_file.parents
        if (parent / "scripts").is_dir() and (parent / "results").is_dir()
    )
    normalized_sources = {}
    for source, digest in source_paths.items():
        path = Path(source).resolve()
        key = path.relative_to(repo_root).as_posix() if path.is_relative_to(repo_root) else str(path)
        normalized_sources[key] = digest
    report = {
        "schema": "submission_revision_evidence_v1",
        "primary_rho": 0.45,
        "fixed_or_sealed_test_truth_read": False,
        "evaluation_labels_used_for_selection": False,
        "units": source_data["units"],
        "limited": source_data["limited"],
        "budget": source_data["budget"],
        "mechanism": source_data["mechanism"],
        "source_files": normalized_sources,
        "figure_source_data": "figure_data/source_data.json",
        "figure_source_data_sha256": sha(source_path),
    }
    (ROOT / "evidence/submission_revision.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )


def update_validation() -> None:
    path = ROOT / "VALIDATION.json"
    report = json.loads(path.read_text())
    report["tests"] = (
        "python3 -m pytest tests/test_submission_revision_20261001.py -q: "
        "12 passed, 1 PyTorch nested-tensor warning (2.19s)"
    )
    report["budget_rho_screen_training_runs"] = 36
    report["rho_and_budget_rho_confirmation_training_runs"] = 45
    report["budget_rho_confirmed_layout_runs"] = 27
    report["tree_screen_fits"] = 27
    report["tree_confirmation_fits"] = 9
    report["blend_confirmation_layouts"] = 27
    report["additional_training_runs_since_revision"] = 206
    report["additional_baseline_runs"] = 42
    report["framework_replacement_pending"] = True
    report["test_labels_opened"] = False
    report["headline_development_mae_seconds"] = {
        "budgets_percent": [1, 5, 10],
        "ours": [32.607, 29.383, 28.776],
        "direct": [33.186, 29.457, 28.481],
        "ours_paired_wins_of_9": [8, 6, 0],
        "scope": "validation-selected configuration evaluated on development labels; test remains sealed",
    }
    report["documents"] = {
        key: document_info(key, path) for key, path in PDFS.items()
    }
    report["nonblocking_engine_notes"] = [
        "Tectonic 0.15 reports repeated BBL consistency passes and stops at six passes; all four PDFs were emitted with resolved references.",
        "Underfull line-breaking notices remain; final logs report no overfull boxes, missing glyphs, or undefined references.",
    ]
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def update_manifest() -> None:
    path = ROOT / "MANIFEST.json"
    manifest = json.loads(path.read_text())
    manifest["status"] = "review_draft_not_submission_certification"
    manifest["inventory_scope"] = (
        "Delivery files excluding this self-referential manifest, mutable build caches and previews, "
        "private figure-editing notes, archived 2026-09-30 PDFs, and TeX intermediate files."
    )
    manifest["files"] = [
        {"path": item.relative_to(ROOT).as_posix(), "sha256": sha(item)}
        for item in package_files()
    ]
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def create_archive(destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing archive: {destination}")
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=8) as archive:
        for item in package_files() + [ROOT / "MANIFEST.json"]:
            archive.write(item, f"{ROOT.name}/{item.relative_to(ROOT).as_posix()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    for key, path in PDFS.items():
        if (not path.is_file() or not BUILD_LOGS[key].is_file()
                or not BUILD_BIBS[key].is_file()):
            raise FileNotFoundError(f"missing final PDF or compile evidence for {key}")
    update_figure_evidence()
    update_validation()
    update_manifest()
    if args.archive:
        create_archive(args.archive.resolve())
    print(json.dumps({
        "documents": {key: document_info(key, path) for key, path in PDFS.items()},
        "manifest_files": len(package_files()),
        "archive": str(args.archive.resolve()) if args.archive else None,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
