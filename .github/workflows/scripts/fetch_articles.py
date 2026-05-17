#!/usr/bin/env python3
"""
Fetch latest articles from JAAD, JAMA Dermatology, and JEADV via PubMed,
then generate Chinese summaries using the Anthropic Claude API.
"""

import json
import os
import time
from datetime import datetime, timedelta, timezone
import requests
import anthropic

# ── Configuration ──────────────────────────────────────────────────────────────

JOURNALS = [
    {
        "name": "JAAD",
        "full_name": "Journal of the American Academy of Dermatology",
        "issn": "0190-9622",
        "color": "#2563eb",
    },
    {
        "name": "JAMA Dermatol",
        "full_name": "JAMA Dermatology",
        "issn": "2168-6068",
        "color": "#16a34a",
    },
    {
        "name": "JEADV",
        "full_name": "Journal of the European Academy of Dermatology and Venereology",
        "issn": "0926-9959",
        "color": "#9333ea",
    },
]

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
MAX_ARTICLES_PER_JOURNAL = 8   # 每本期刊最多抓幾篇
DAYS_BACK = 3                  # 抓幾天內的新文章（週末補抓）
OUTPUT_PATH = "docs/data/articles.json"

# ── PubMed helpers ─────────────────────────────────────────────────────────────

def build_date_range() -> tuple[str, str]:
    today = datetime.now(timezone.utc)
    start = today - timedelta(days=DAYS_BACK)
    return start.strftime("%Y/%m/%d"), today.strftime("%Y/%m/%d")


def search_pubmed(issn: str, min_date: str, max_date: str) -> list[str]:
    """Return a list of PubMed IDs for the given ISSN and date range."""
    params = {
        "db": "pubmed",
        "term": f"{issn}[ISSN] AND {min_date}:{max_date}[PDAT]",
        "retmax": MAX_ARTICLES_PER_JOURNAL,
        "retmode": "json",
        "sort": "pub+date",
    }
    r = requests.get(f"{PUBMED_BASE}/esearch.fcgi", params=params, timeout=15)
    r.raise_for_status()
    return r.json().get("esearchresult", {}).get("idlist", [])


def fetch_details(pmids: list[str]) -> list[dict]:
    """Fetch article details for a list of PMIDs."""
    if not pmids:
        return []
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "json",
        "rettype": "abstract",
    }
    r = requests.get(f"{PUBMED_BASE}/efetch.fcgi", params=params, timeout=20)
    r.raise_for_status()

    # PubMed JSON structure
    articles = []
    data = r.json()
    result = data.get("PubmedArticleSet", {})

    # Sometimes the key is wrapped differently; use summary endpoint for reliability
    params2 = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "json",
    }
    r2 = requests.get(f"{PUBMED_BASE}/esummary.fcgi", params=params2, timeout=20)
    r2.raise_for_status()
    summaries = r2.json().get("result", {})

    for pmid in pmids:
        item = summaries.get(pmid)
        if not item or item.get("uid") != pmid:
            continue
        authors = [a.get("name", "") for a in item.get("authors", [])[:3]]
        author_str = ", ".join(authors)
        if len(item.get("authors", [])) > 3:
            author_str += " et al."
        articles.append({
            "pmid": pmid,
            "title": item.get("title", "").rstrip("."),
            "authors": author_str,
            "pub_date": item.get("pubdate", ""),
            "doi": next(
                (aid["value"] for aid in item.get("articleids", []) if aid["idtype"] == "doi"),
                "",
            ),
            "abstract": "",  # will be filled below
        })

    # Fetch abstracts separately via efetch (XML → parse text)
    if articles:
        abs_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
        }
        abs_r = requests.get(f"{PUBMED_BASE}/efetch.fcgi", params=abs_params, timeout=20)
        abs_r.raise_for_status()
        # Simple regex-free extraction
        xml = abs_r.text
        for art in articles:
            tag = f"<PMID Version=\"1\">{art['pmid']}</PMID>"
            start = xml.find(tag)
            if start == -1:
                continue
            chunk = xml[start: start + 8000]
            abs_start = chunk.find("<AbstractText")
            abs_end = chunk.find("</Abstract>")
            if abs_start != -1 and abs_end != -1:
                raw = chunk[abs_start: abs_end]
                # Strip XML tags
                import re
                art["abstract"] = re.sub(r"<[^>]+>", " ", raw).strip()

    return articles


# ── Claude summarization ───────────────────────────────────────────────────────

def summarize_article(client: anthropic.Anthropic, article: dict) -> str:
    """Generate a concise Traditional Chinese summary using Claude."""
    abstract_text = article["abstract"] or "(no abstract available)"
    prompt = f"""你是一位皮膚科醫師助理，請以繁體中文，用條列式撰寫以下皮膚科論文的重點摘要。

格式要求：
- 研究目的（1-2句）
- 主要方法（1句）
- 重要結果（2-3點）
- 臨床意義（1-2句）

論文標題：{article['title']}
摘要原文：{abstract_text}

請直接輸出摘要，不需要任何前言。"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set.")
    client = anthropic.Anthropic(api_key=api_key)

    min_date, max_date = build_date_range()
    print(f"Fetching articles from {min_date} to {max_date}")

    # Load existing data to avoid re-summarizing
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = {"updated_at": "", "journals": {}}

    seen_pmids = set()
    for journal_data in existing.get("journals", {}).values():
        for art in journal_data.get("articles", []):
            seen_pmids.add(art["pmid"])

    output = {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "date_range": f"{min_date} ~ {max_date}",
        "journals": {},
    }

    for journal in JOURNALS:
        print(f"\n── {journal['name']} ──")
        pmids = search_pubmed(journal["issn"], min_date, max_date)
        print(f"  Found {len(pmids)} articles: {pmids}")

        new_pmids = [p for p in pmids if p not in seen_pmids]
        articles = fetch_details(new_pmids)

        # Carry over previous articles for this journal (keep last 30)
        prev_articles = existing.get("journals", {}).get(journal["name"], {}).get("articles", [])
        prev_pmids_set = {a["pmid"] for a in prev_articles}

        summarized = []
        for art in articles:
            if art["pmid"] in prev_pmids_set:
                continue
            print(f"  Summarizing: {art['title'][:60]}...")
            try:
                art["summary_zh"] = summarize_article(client, art)
            except Exception as e:
                print(f"  Warning: summarization failed: {e}")
                art["summary_zh"] = "（摘要生成失敗，請見原文）"
            summarized.append(art)
            time.sleep(1)  # rate-limit courtesy

        combined = summarized + prev_articles
        combined = combined[:30]  # keep max 30 per journal

        output["journals"][journal["name"]] = {
            "full_name": journal["full_name"],
            "color": journal["color"],
            "articles": combined,
        }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
