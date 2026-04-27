import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def paraphrase_en(q: str, idx: int) -> str:
    text = " ".join((q or "").strip().split())
    if not text:
        return text

    replacements = [
        ("What is ", "Can you explain "),
        ("How do I ", "What's the best way to "),
        ("How can I ", "What is the process to "),
        ("Can I ", "Is it possible to "),
        ("Does ", "Could you clarify whether "),
        ("Which ", "What are the best options for "),
    ]
    for src, tgt in replacements:
        if text.startswith(src):
            text = tgt + text[len(src):]
            break

    if idx % 3 == 0 and " in Pakistan" not in text:
        text = text.rstrip("?") + " in Pakistan?"
    elif idx % 3 == 1:
        text = text.rstrip("?") + "?"
    else:
        text = "For Pakistan: " + text.rstrip("?") + "?"

    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/splits/test.jsonl")
    parser.add_argument("--output", default="retrieval/eval/english_paraphrase_test.jsonl")
    parser.add_argument("--n", type=int, default=40)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    selected = [r for r in rows if r.get("query_en")]
    selected = selected[: args.n]

    out_rows = []
    for i, r in enumerate(selected):
        out_rows.append(
            {
                "id": r.get("id"),
                "query": paraphrase_en(r.get("query_en", ""), i),
                "language": "english",
                "benchmark": "english_paraphrase_v1",
            }
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(out_rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
