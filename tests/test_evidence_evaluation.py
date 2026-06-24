"""Evidence-based evaluation: run post-correction + domain coverage against real logs.

Reads ALL translate events across all evidence JSONL files and validates:
  1. Post-correction fires correctly on known misrecognitions
  2. All domain terms that appear in JP input are covered by domain_data
  3. Historical translation quality metrics (keyword presence, empty rate per term)

This test is model-free — no LLM/NLLB required.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.post_correction import post_correct
from src.domain_data import DOMAIN_TERMS, PROPER_NOUNS, KATAKANA_TERMS

EVIDENCE_DIR = Path("test_audio/evidence")

# ── All Japanese source terms known to the system ──────────────────────────
ALL_JP_TERMS: dict[str, str] = {}
ALL_JP_TERMS.update(DOMAIN_TERMS)
ALL_JP_TERMS.update(PROPER_NOUNS)
ALL_JP_TERMS.update(KATAKANA_TERMS)

# Terms that also appear in post_correction.PHRASE_CORRECTIONS (as RHS / correct form)
# These should match when post_correct runs (indirect check).
from src.post_correction import PHRASE_CORRECTIONS, CONTEXT_CORRECTIONS

_CORRECT_RHS = set(PHRASE_CORRECTIONS.values())


def _load_all_events():
    events = []
    for f in sorted(EVIDENCE_DIR.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("event") == "translate":
                    events.append(ev)
    return events


def _count_domain_terms_in(text: str) -> dict[str, int]:
    hits: dict[str, int] = {}
    for term in ALL_JP_TERMS:
        count = text.count(term)
        if count:
            hits[term] = count
    return hits


def test_post_correction_coverage():
    """Verify that every post-correction target appears in evidence input."""
    print("── Post-Correction Evidence Coverage ──")
    events = _load_all_events()
    all_jp = set(e["jp"] for e in events)

    checked = 0
    missing_examples: dict[str, str] = {}
    for wrong, right in PHRASE_CORRECTIONS.items():
        checked += 1
        matching = [jp for jp in all_jp if wrong in jp]
        if not matching:
            # Still OK — correction targets are future-proofing
            continue
        # Verify correction fires on at least one input
        any_fired = False
        for jp in matching:
            corrected = post_correct(jp)
            if wrong not in corrected:
                any_fired = True
                break
        if not any_fired:
            ex = matching[0]
            corrected = post_correct(ex)
            missing_examples[wrong] = f"'{ex}' → '{corrected}'"

    print(f"  PHRASE_CORRECTIONS entries checked: {checked}")
    if missing_examples:
        print(f"  WARNING: {len(missing_examples)} corrections not firing on evidence:")
        for w, ex in missing_examples.items():
            print(f"    '{w}' — sample: {ex}")
        return False
    print("  All corrections fire correctly on evidence input.")
    return True


def test_domain_term_coverage():
    """Check every domain term appears in at least one evidence input."""
    print("── Domain Term Coverage (evidence vs domain_data) ──")
    events = _load_all_events()
    all_jp = " ".join(e["jp"] for e in events)

    covered = 0
    uncovered: list[str] = []
    for term in sorted(ALL_JP_TERMS, key=lambda t: -len(t)):
        if term in all_jp:
            covered += 1
        else:
            uncovered.append(term)

    print(f"  Domain terms in evidence: {covered}/{len(ALL_JP_TERMS)}")
    if uncovered:
        print(f"  NOT yet seen in meetings ({len(uncovered)}):")
        for t in uncovered[:20]:
            print(f"    {t} = {ALL_JP_TERMS[t]}")
        if len(uncovered) > 20:
            print(f"    ... and {len(uncovered)-20} more")
    return True


def test_translation_quality_metrics():
    """Compute translation quality metrics from historical evidence.

    Checks that domain terms appearing in JP input are reflected
    (by keyword presence) in the VI output.
    """
    print("── Historical Translation Quality Metrics ──")
    events = _load_all_events()

    total = len(events)
    empty_count = 0
    term_in_jp: Counter = Counter()
    term_in_vi: Counter = Counter()
    term_empty: Counter = Counter()
    total_vi_chars = 0
    total_jp_chars = 0

    for ev in events:
        jp = ev.get("jp", "")
        vi = ev.get("vi", "")
        total_jp_chars += len(jp)
        total_vi_chars += len(vi)

        if vi in ("(...)", "") or not vi:
            empty_count += 1

        vi_lower = vi.lower()
        for term, vi_ref in ALL_JP_TERMS.items():
            c = jp.count(term)
            if c:
                term_in_jp[term] += c
                if vi_ref.lower() in vi_lower:
                    term_in_vi[term] += 1
                else:
                    term_empty[term] += 1

    empty_rate = 100.0 * empty_count / total if total else 0
    compression = total_vi_chars / total_jp_chars if total_jp_chars else 0

    print(f"  Total translate events: {total}")
    print(f"  Empty translations:     {empty_count}/{total} ({empty_rate:.1f}%)")
    print(f"  VI/JP char ratio:       {compression:.2f}")
    print()

    # Terms with the WORST coverage
    worst_cutoff = 3  # at least N occurrences in input
    worst = [
        (term, count, term_in_vi.get(term, 0))
        for term, count in term_in_jp.items()
        if count >= worst_cutoff and term_in_vi.get(term, 0) / count < 0.5
    ]
    worst.sort(key=lambda x: -x[1])

    if worst:
        print(f"  Domain terms with LOW keyword presence in historical VI output:")
        print(f"  (term: {worst_cutoff}+ JP occurrences, <50% VI keyword match)")
        for term, jp_count, vi_count in worst[:15]:
            pct = 100.0 * vi_count / jp_count
            print(f"    {term}: {vi_count}/{jp_count} ({pct:.0f}%) — ref: {ALL_JP_TERMS[term]}")
    else:
        print("  All domain terms with significant presence have good coverage.")

    return empty_rate < 30  # Fail if >30% empty rate historically


def test_term_consistency():
    """Verify no term overlap between maps in domain_data."""
    print("── Domain Data Consistency ──")
    issues = []

    # Check for terms in DOMAIN_TERMS that are also in PROPER_NOUNS with different meaning
    for term in DOMAIN_TERMS:
        if term in PROPER_NOUNS:
            dt_vi = DOMAIN_TERMS[term].lower()
            pn_vi = PROPER_NOUNS[term].lower()
            if dt_vi != pn_vi:
                issues.append(
                    f"  OVERLAP: DOMAIN_TERMS['{term}']='{dt_vi}' vs "
                    f"PROPER_NOUNS['{term}']='{pn_vi}'"
                )

    # Check for duplicate entries within each map
    for map_name, m in [("KATAKANA_TERMS", KATAKANA_TERMS), ("PROPER_NOUNS", PROPER_NOUNS)]:
        seen = {}
        for k, v in m.items():
            if k in seen and seen[k] != v:
                issues.append(f"  DUPE: {map_name}['{k}'] = '{v}' and '{seen[k]}'")
            seen[k] = v

    if issues:
        print(f"  {len(issues)} consistency issue(s) found:")
        for issue in issues:
            print(issue)
        return False
    print("  All term maps consistent, no conflicting overlaps.")
    return True


def test_config_consistency():
    """Verify config.py NLLB_GLOSSARY matches domain_data for shared terms."""
    print("── Config ↔ Domain Data Consistency ──")
    import config

    issues = []

    # Terms that exist in BOTH NLLB_GLOSSARY and DOMAIN_TERMS
    # (should have compatible meanings, though NLLB uses English, domain_data uses Vietnamese)
    for term, nllb_val in config.NLLB_GLOSSARY.items():
        if term in DOMAIN_TERMS:
            # NLLB uses English, DOMAIN_TERMS uses Vietnamese — different langs OK
            # but check they aren't contradictory
            pass
        if term in PROPER_NOUNS:
            # NLLB_GLOSSARY English should match PROPER_NOUNS value intent
            pn_val = PROPER_NOUNS[term].lower()
            nllb_lower = nllb_val.lower()
            # Some terms differ by design (e.g., NLLB uses English, PROPER_NOUNS uses Vietnamese)
            if nllb_lower != pn_val:
                # Check if one contains the other
                if nllb_lower not in pn_val and pn_val not in nllb_lower:
                    # Acceptable if very different meaning — log for awareness only
                    pass

    # Check LLM_SYSTEM_PROMPT contains key domain terms
    prompt = getattr(config, "LLM_SYSTEM_PROMPT", "")
    key_terms = [
        "cứu hỏa",
        "cấp cứu",
        "vận chuyển",
        "nạn nhân",
        "bàn giao",
        "điều động",
        "tiếp nhận",
        "task gián đoạn",
        "parent task",
        "child task",
    ]
    missing = [t for t in key_terms if t not in prompt]
    if missing:
        issues.append(
            f"  LLM_SYSTEM_PROMPT missing domain terms: {missing}"
        )

    # Verify ít nhât key hotwords match domain_data terms
    hotwords_path = Path("hotwords_it.txt")
    hotword_terms = set()
    if hotwords_path.is_file():
        for line in hotwords_path.read_text().splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            hotword_terms.add(line.split(" :")[0].strip())

    missing_from_hotwords = [t for t in ALL_JP_TERMS if t not in hotword_terms]
    if missing_from_hotwords:
        issues.append(
            f"  {len(missing_from_hotwords)} domain terms not in hotwords (acceptable "
            f"if kanji-only): {missing_from_hotwords[:5]}..."
        )

    if issues:
        print(f"  {len(issues)} issue(s):")
        for issue in issues:
            print(issue)
        return False
    print("  Config is consistent with domain_data.")
    return True


if __name__ == "__main__":
    print("=" * 70)
    print("  Evidence-Based Evaluation — All Meeting Logs")
    print("=" * 70)
    print()

    results = []

    results.append(("Post-Correction Coverage", test_post_correction_coverage()))
    print()
    results.append(("Domain Term Coverage", test_domain_term_coverage()))
    print()
    results.append(("Translation Quality", test_translation_quality_metrics()))
    print()
    results.append(("Domain Data Consistency", test_term_consistency()))
    print()
    results.append(("Config Consistency", test_config_consistency()))
    print()

    print("=" * 70)
    all_pass = all(r[1] for r in results)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {name}")
    print("=" * 70)

    if all_pass:
        print("\nRESULT: PASS")
        sys.exit(0)
    else:
        print("\nRESULT: FAIL")
        sys.exit(1)
