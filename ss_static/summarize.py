from __future__ import annotations

import sys

sys.dont_write_bytecode = True
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

import argparse
import csv
from datetime import datetime
from pathlib import Path


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fmt(v, nd=4) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "—" if v in (None, "") else str(v)


def build_summary(archive_root, project_root=None) -> Path:
    root = (Path(archive_root) if archive_root is not None
            else Path(project_root or ".").resolve() / "results")
    base = root / "ss_static"
    rows = _read_csv(base / "index.csv")
    evals = [r for r in rows if r.get("kind") == "eval"
             and r.get("status") == "completed"]
    calibs = [r for r in rows if r.get("kind") == "calib"
              and r.get("status") == "completed"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = base / f"SUMMARY_{ts}.md"
    i = 1
    while out.exists():
        out = base / f"SUMMARY_{ts}_{i}.md"
        i += 1
    lines = [
        f"# (s,S) 对比验证跨运行总结",
        "",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 索引：`{base / 'index.csv'}`",
        f"- 已完成评估运行：{len(evals)}；已完成标定运行：{len(calibs)}",
        "",
        "## 1. 主结果（测试集配对）",
        "",
    ]
    if evals:
        lines += [
            "| run_id | T | 路径 | 算法 | 起点 | 策略均值 | 基线均值 | 配对差 | 相对差 | 胜率 | 审计 |",
            "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---|",
        ]
        for r in evals:
            audit = "通过" if str(r.get("audit_ok", "")) in ("1", "True", "true") else "—"
            lines.append(
                f"| {r.get('run_id')} | {r.get('T')} | {r.get('n_paths')} | "
                f"{r.get('algorithm', '—')} | {r.get('selected_start', '—')} | "
                f"{_fmt(r.get('test_cost'))} | {_fmt(r.get('baseline_mean'))} | "
                f"{_fmt(r.get('diff_mean'))} | {_fmt(r.get('rel_pct'), 2)}% | "
                f"{_fmt(float(r.get('win_rate', 0)) * 100, 2)}% | {audit} |")
    else:
        lines.append("- 暂无完成的评估运行。")
    lines += ["", "## 2. 标定/算法对比", ""]
    if calibs:
        for r in calibs:
            d = base / str(r.get("run_id"))
            alg = _read_csv(d / "algorithm_summary.csv")
            lines += [
                f"### `{r.get('run_id')}`（T={r.get('T')}，"
                f"{r.get('n_paths')} 条训练路径）",
                "",
            ]
            if alg:
                has_h = any(str(a.get("h", "")).strip() for a in alg)
                hdr = ("| 算法 | 起点 | s | S | h | 训练成本 | 验证成本 | n_unique | 用时 s | 选中 |"
                       if has_h else
                       "| 算法 | 起点 | s | S | 训练成本 | 验证成本 | n_unique | 用时 s | 选中 |")
                sep = ("|---|---|---|---|---|---:|---:|---:|---:|---|"
                       if has_h else
                       "|---|---|---|---|---:|---:|---:|---:|---|")
                lines += [hdr, sep]
                for a in alg:
                    hcell = f" {a.get('h')} |" if has_h else ""
                    lines.append(
                        f"| {a.get('algorithm')} | {a.get('start')} | "
                        f"{a.get('s')} | {a.get('S')} |{hcell} "
                        f"{_fmt(a.get('train_cost'))} | {_fmt(a.get('val_cost'))} | "
                        f"{a.get('n_unique')} | {_fmt(a.get('wall_s'), 1)} | "
                        f"{'✓' if str(a.get('selected')) == '1' else ''} |")
            else:
                lines.append("- 缺少 `algorithm_summary.csv`。")
            lines.append("")
    else:
        lines.append("- 暂无完成的标定运行。")
    lines += [
        "",
        "---",
        "",
        "> 详细口径、五项成本、行为、时间剖面、期末值敏感性、审计见每个 run 的",
        "> `summary.md`（评估）/`calibration.md`（标定）。本文件只做跨运行索引。",
        "> 成本单位万元；不同 T/不同路径数的运行不要直接横向比较。",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser(description="生成跨运行 (s,S) 对比总结")
    ap.add_argument("--archive-root", default=None,
                    help="归档根；缺省 <项目>/results")
    ap.add_argument("--project-root", default=None)
    args = ap.parse_args()
    out = build_summary(args.archive_root, args.project_root)
    print(f"summary: {out}")


if __name__ == "__main__":
    main()
