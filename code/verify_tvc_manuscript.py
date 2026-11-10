#!/usr/bin/env python3
"""TVC 投稿前自检：数字可回溯、引用一致、格式硬项。

用法: python verify_tvc_manuscript.py <manuscript.md> [--runs runs.csv]

检查项:
 A. 摘要词数 150-250；关键词 4-6
 B. [TBD]/[REF-] 残留
 C. 引用编号: 正文最大引用号 = 文献条数, 首引顺序升序, 无重复条目, 无未引条目
 D. 表: caption 数 vs in-text 引用; 图: caption 数 vs in-text 引用
 E. 数字可回溯: 抽取正文中形如 28.848 / -0.852 / 810,585 的关键数字, 标注是否出现在 runs.csv（仅提示, 不强制）
 F. TVC 格式项: Declarations 五要素、Data availability、Keywords、作者占位
"""
import argparse
import csv
import os
import re
import sys


def load_manuscript(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def split_body_refs(t):
    idx = t.rfind("## References")
    if idx < 0:
        return t, ""
    return t[:idx], t[idx:]


def check_abstract(t):
    m = re.search(r"## Abstract\s*\n\n(.+?)\n\n", t, re.S)
    if not m:
        return ["Abstract 未找到"], {}
    words = len(m.group(1).split())
    msgs = []
    if not (150 <= words <= 250):
        msgs.append(f"Abstract {words} words (need 150-250)")
    kw = re.search(r"\*\*Keywords\*\*:\s*(.+)", t)
    kw_n = len([k for k in kw.group(1).split(";") if k.strip()]) if kw else 0
    if not (4 <= kw_n <= 6):
        msgs.append(f"Keywords {kw_n} (need 4-6)")
    return msgs, {"abstract_words": words, "keywords": kw_n}


def check_residue(t, body):
    msgs = []
    for pat, name in [(r"\[TBD", "TBD"), (r"\[REF-", "REF-placeholder")]:
        n = len(re.findall(pat, body))
        if n:
            msgs.append(f"{name} residue in body: {n}")
    return msgs, {}


def expand_cites(s):
    """Expand '[2,3]', '[5-7]', '[2, 3]' -> [2,3,5,6,7] (order preserved)."""
    out = []
    for tok in re.split(r",\s*", s):
        tok = tok.strip()
        m = re.match(r"(\d+)\s*[-–]\s*(\d+)$", tok)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            out.extend(range(a, b + 1))
        elif tok.isdigit():
            out.append(int(tok))
    return out


def check_citations(body, refs):
    msgs = []
    cites = []
    for m in re.finditer(r"\[([\d,\-–\s]+)\]", body):
        cites.extend(expand_cites(m.group(1)))
    ref_nums = [int(x) for x in re.findall(r"^\[(\d+)\] ", refs, re.M)]
    if not ref_nums:
        return ["参考文献列表解析失败"], {}
    maxc = max(cites) if cites else 0
    if maxc != len(ref_nums):
        msgs.append(f"max in-text cite {maxc} != reference count {len(ref_nums)}")
    # first-citation order
    seen, order, bad = set(), [], []
    for c in cites:
        if c not in seen:
            seen.add(c)
            order.append(c)
    expected = list(range(1, len(ref_nums) + 1))
    if order != expected:
        diff = [i for i, (a, b) in enumerate(zip(order, expected)) if a != b]
        first_bad = order[diff[0]] if diff else None
        msgs.append(f"first-citation order not 1..N (first mismatch at pos {diff[0] if diff else '?'}, got {first_bad})")
    # duplicates by title similarity (simple: identical first 60 chars of first author block)
    entries = re.findall(r"^\[(\d+)\] (.+)$", refs, re.M)
    texts = [e[1][:80].lower() for e in entries]
    dups = [texts[i] for i in range(len(texts)) if texts[i] in texts[:i]]
    if dups:
        msgs.append(f"duplicate reference entries: {dups}")
    unused = [n for n in ref_nums if n not in cites]
    if unused:
        msgs.append(f"references never cited: {unused}")
    return msgs, {"n_refs": len(ref_nums), "max_cite": maxc}


def check_tables_figs(body):
    msgs = []
    tab_caps = sorted(set(int(x) for x in re.findall(r"\*\*Table\s*(\d+)\*\*", body)))
    tab_refs = sorted(set(int(x) for x in re.findall(r"Table\s*(\d+)", body)))
    # refs include captions; remove caption occurrences by counting mentions outside caption lines
    lines = body.split("\n")
    noncap_refs = set()
    for ln in lines:
        if ln.strip().startswith("**Table"):
            continue
        for x in re.findall(r"Table\s*(\d+)", ln):
            noncap_refs.add(int(x))
    missing_ref = [x for x in tab_caps if x not in noncap_refs]
    if missing_ref:
        msgs.append(f"tables without in-text citation: {missing_ref}")
    fig_caps = sorted(set(int(x) for x in re.findall(r"\*\*Fig(?:ure)?\.?\s*(\d+)", body)))
    fig_refs = set()
    for ln in lines:
        if ln.strip().startswith("**Fig"):
            continue
        for x in re.findall(r"Fig(?:ure)?\.?\s*(\d+)", ln):
            fig_refs.add(int(x))
    if fig_caps:
        missing_fig = [x for x in fig_caps if x not in fig_refs]
        if missing_fig:
            msgs.append(f"figures without in-text citation: {missing_fig}")
    return msgs, {"tables": tab_caps, "figs": fig_caps}


def check_declarations(t):
    msgs = []
    title_page = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "submission_tvc", "title-page.md")
    if os.path.exists(title_page):
        tp = open(title_page, encoding="utf-8").read()
        if not any(k in tp for k in ("Author biographies", "Author biography")):
            msgs.append("title page is missing the author biographies block")
    else:
        msgs.append("title page (submission_tvc/title-page.md) not found")
    required = [
        "Statements and Declarations",
        "Funding",
        "Competing interests",
        "Ethics approval",
        "Data availability",
        "Code availability",
        "Author contributions",
        "Use of AI-assisted tools",
        "Acknowledgments",
        "ORCID",
    ]
    for key in required:
        variants = [key, key.replace("Author contributions", "Authors' contributions")]
        if not any(v.lower() in t.lower() for v in variants):
            msgs.append(f"missing declaration/title-page item: {key}")
    # TVC artwork guidance: captions must not end with a full stop
    for m in re.finditer(r"\*\*Fig\. (\d+)\*\* ([^\n]+)$", t, re.M):
        if m.group(2).rstrip().endswith("."):
            msgs.append(f"Fig. {m.group(1)} caption ends with a full stop (TVC guidance)")
    return msgs, {}


def load_runs(path):
    try:
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
    except Exception:
        return []
    vals = set()
    for r in rows:
        for k in ("psnr_test", "gaussians"):
            v = (r.get(k) or "").strip()
            if v and v not in ("n/a", ""):
                try:
                    vals.add(float(v))
                except ValueError:
                    pass
    return rows, vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manuscript")
    ap.add_argument("--runs", default=r"D:\3dgs\proj-D6\runs.csv")
    a = ap.parse_args()
    t = load_manuscript(a.manuscript)
    body, refs = split_body_refs(t)
    report, stats = [], {}
    for fn in (lambda: check_abstract(t), lambda: check_residue(t, body),
               lambda: check_citations(body, refs), lambda: check_tables_figs(body),
               lambda: check_declarations(t)):
        msgs, st = fn()
        report += msgs
        stats.update(st)
    print("== checks ==")
    if report:
        for m in report:
            print("FAIL:", m)
    else:
        print("PASS: all automated checks")
    print("== stats ==")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    # numeric traceability (informational)
    rows_vals = load_runs(a.runs)
    if rows_vals:
        rows, vals = rows_vals
        nums = set()
        for m in re.findall(r"(?<![\w.])(\d{2,3}\.\d{3})(?![\d])", body):
            nums.add(float(m))
        missing = sorted(x for x in nums if x not in vals)
        print(f"  PSNR-like numbers in body: {len(nums)}, not found in runs.csv: {len(missing)}")
        if missing[:10]:
            print("   e.g.", missing[:10])
        print(f"  runs.csv rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
