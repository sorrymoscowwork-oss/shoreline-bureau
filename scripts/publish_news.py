#!/usr/bin/env python3
"""SHORELINE BUREAU editorial publisher.

Uses Gemini Interactions API with Google Search grounding, validates a multilingual
article against the existing blog schema, then commits the article to the repo.
No external Python packages are required.
"""
from __future__ import annotations

import argparse
import json
import html
import os
import re
import subprocess
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
ARTICLES_FILE = ROOT / "blog" / "articles.json"
ARTICLE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="theme-color" content="#fff">
  <meta name="description" content="SHORELINE BUREAU editorial — Hurghada and the Red Sea.">
  <title>Editorial — SHORELINE BUREAU</title>
  <link rel="stylesheet" href="/assets/blog.css?v=20261121-test">
  <link rel="stylesheet" href="/assets/site-header.css?v=20261122-shared-base">
  <script src="/assets/site-header.js?v=20261121-header-match" defer></script>
</head>
<body>
  <div id="site-header-root"></div>
  <div id="article-root"><main class="blog-page"><p class="blog-empty">Loading article…</p></main></div>
  <script src="/assets/article.js?v=20261121-test" defer></script>
  <script>document.documentElement.dataset.article="__SLUG__";</script>
</body>
</html>
"""
LANGUAGES = ("en", "ru", "de", "zh")
API_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
MODEL = "gemini-3.8-flash"


def log(message: str) -> None:
    print(f"[SHORELINE] {message}", flush=True)


def gemini_request(prompt: str) -> dict:
    """Gemini 3.8 Flash via the free-tier Interactions API. No paid fallback."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GitHub secret GEMINI_API_KEY is not available to this workflow.")
    payload = {"model": MODEL, "input": prompt}
    req = Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
    )
    try:
        with urlopen(req, timeout=300) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1600]
        if exc.code == 429:
            raise RuntimeError("Gemini free-tier quota/rate limit reached (HTTP 429). No paid fallback is configured; nothing was published. " + detail) from exc
        raise RuntimeError(f"Gemini Interactions API returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Gemini API: {exc.reason}") from exc
    if data.get("status") not in (None, "completed"):
        raise RuntimeError(f"Gemini interaction did not complete: {data.get('status')}")
    return data


def create_interaction(prompt: str) -> dict:
    return gemini_request(prompt)


def extract_output(response: dict) -> tuple[str, list[dict]]:
    if response.get("output_text"):
        return response["output_text"].strip(), []
    chunks = []
    for step in response.get("steps", []):
        if step.get("type") == "model_output":
            for part in step.get("content", []):
                if part.get("type") == "text" and part.get("text"):
                    chunks.append(part["text"])
    return "\n".join(chunks).strip(), []


def parse_json(text: str) -> dict:
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text, flags=re.I)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError("Gemini did not return a JSON article.")
    try:
        return json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Gemini returned invalid article JSON: {exc}") from exc


def article_prompt(existing: dict, today: str) -> str:
    recent = existing.get("articles", [])[:30]
    archive = []
    for item in recent:
        archive.append({
            "slug": item.get("slug"),
            "date": item.get("date"),
            "titles": {k: v.get("title") for k, v in item.get("lang", {}).items()},
            "sources": [s.get("url") for s in item.get("sources", [])],
        })
    return f"""
You are the independent editorial research desk of SHORELINE BUREAU, an independent
property and place advisory based in Hurghada, Egypt. Today's date in Africa/Cairo is {today}.

TASK
Research Google Search and select ONE genuinely newsworthy, recent, verifiable development
from Hurghada or Egypt's Red Sea region. Relevant beats include: everyday life for tourists
and foreign residents; property and new developments; developers and project launches;
investment; Sahl Hasheesh, Soma Bay, El Gouna, Makadi Bay, Safaga, Marsa Alam and other
Red Sea districts; infrastructure, roads, utilities, healthcare, education, ecology,
culture, airports, flight routes, tourism, immigration/residency policy, and local prices.

Choose a concrete event or development, not a generic trend. Prefer a primary source
(government, developer, airport, airline, operator, official announcement) plus at least
two independent/reputable reports where possible. Focus on news published or materially
updated in the last 45 days. Verify dates and distinguish announcements, plans, construction
and completed facilities. Never invent project names, prices, dates, permits, quotes or
statistics. If reliable fresh news cannot be found, return exactly:
{{"no_publish":true,"reason":"No sufficiently verified recent local development found."}}

EDITORIAL STANDARD
Write a substantial article, approximately two A4 pages: target 850–1,100 words in English
across 5–7 sections. Explain what happened, who is involved, location, status and timeline;
why it matters to people living in or visiting the Red Sea region; likely implications for
property buyers/investors where relevant; what remains uncertain and what readers should
verify. Include a measured SHORELINE BUREAU perspective as the final section. Be analytical,
independent and specific. No promotional tone, no invented advice, no unsupported certainty.
Clearly label forward-looking statements as expectations, not facts.

TRANSLATIONS
Produce complete, natural editorial versions in English (en), Russian (ru), German (de),
and Simplified Chinese (zh). These are full translations/adaptations of the same facts and
analysis, not summaries. Preserve names, dates, numbers and caveats consistently.

DUPLICATE CONTROL
Do not repeat any of these recently published articles or their source stories:
{json.dumps(archive, ensure_ascii=False)}

OUTPUT
Return ONLY valid JSON (no Markdown fences) with this exact structure:
{{
  "no_publish": false,
  "slug": "short-ascii-kebab-case-slug",
  "category": {{
    "en": "Category",
    "ru": "Категория",
    "de": "Kategorie",
    "zh": "分类"
  }},
  "lang": {{
    "en": {{"title":"...", "excerpt":"...", "sections":[["Heading","Paragraph"], ...]}},
    "ru": {{"title":"...", "excerpt":"...", "sections":[["Заголовок","Текст"], ...]}},
    "de": {{"title":"...", "excerpt":"...", "sections":[["Überschrift","Text"], ...]}},
    "zh": {{"title":"...", "excerpt":"...", "sections":[["标题","正文"], ...]}}
  }}
}}
Use 5–7 sections per language. The final section must be the SHORELINE BUREAU assessment.
The source links are collected separately from Google's grounding citations; do not fabricate
URLs or put sources into the article text.
"""


def clean_slug(value: str, fallback: str) -> str:
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return (value[:72].strip("-") or fallback)


def collect_sources(citations: list[dict]) -> list[dict]:
    seen = set()
    sources = []
    for citation in citations:
        url = citation["url"].strip()
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc:
            continue
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
        if normalized in seen:
            continue
        seen.add(normalized)
        title = citation.get("title") or parsed.netloc
        sources.append({"label": title[:180], "url": url})
    return sources[:8]


def validate_article(data: dict, sources: list[dict]) -> None:
    if data.get("no_publish"):
        raise NoPublish(data.get("reason", "No verified news selected."))
    if len(sources) < 2:
        raise RuntimeError("Fewer than two verifiable Google Search citations were returned; refusing to publish.")
    if not isinstance(data.get("category"), dict) or not isinstance(data.get("lang"), dict):
        raise RuntimeError("Article is missing category or language data.")
    for language in LANGUAGES:
        category = data["category"].get(language)
        content = data["lang"].get(language)
        if not category or not isinstance(content, dict):
            raise RuntimeError(f"Missing {language} category or article.")
        if not content.get("title") or not content.get("excerpt"):
            raise RuntimeError(f"Missing {language} title or excerpt.")
        sections = content.get("sections")
        if not isinstance(sections, list) or not 5 <= len(sections) <= 7:
            raise RuntimeError(f"{language} article must contain 5–7 sections.")
        for section in sections:
            if not isinstance(section, list) or len(section) != 2 or not all(str(x).strip() for x in section):
                raise RuntimeError(f"Malformed section in {language} article.")
    english_words = sum(len(re.findall(r"\b[\w’'-]+\b", p)) for _, p in data["lang"]["en"]["sections"])
    if english_words < 700:
        raise RuntimeError(f"English article is too short ({english_words} words; minimum 700).")


class NoPublish(Exception):
    pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Generate and validate, but do not commit.")
    args = parser.parse_args()

    today = datetime.now(ZoneInfo("Africa/Cairo")).date().isoformat()
    existing = json.loads(ARTICLES_FILE.read_text(encoding="utf-8"))
    log(f"Starting editorial research for {today}; dry_run={args.dry_run}")
    interaction = create_interaction(article_prompt(existing, today))
    output, citations = extract_output(interaction)
    if not output:
        raise RuntimeError("Gemini interaction returned no text output.")
    article = parse_json(output)
    if article.get("no_publish"):
        log(f"Skipping publication: {article.get('reason', 'No verified story found.')}")
        return 0

    sources = collect_sources(citations)
    validate_article(article, sources)
    slug = clean_slug(article.get("slug", ""), "red-sea-update")
    known = existing.get("articles", [])
    known_slugs = {x.get("slug") for x in known}
    if slug in known_slugs:
        slug = f"{slug}-{today}"
    source_urls = {s["url"] for s in sources}
    for old in known:
        old_urls = {s.get("url") for s in old.get("sources", [])}
        if source_urls & old_urls:
            raise RuntimeError(f"Duplicate source story detected; overlaps existing article {old.get('slug')}.")
    # article.js renders these fields into innerHTML. Escape generated text and
    # source attributes before storing them to prevent model output becoming markup.
    safe_lang = {}
    for language in LANGUAGES:
        content = article["lang"][language]
        safe_lang[language] = {
            "title": html.escape(content["title"], quote=False),
            "excerpt": html.escape(content["excerpt"], quote=False),
            "sections": [
                [html.escape(section[0], quote=False), html.escape(section[1], quote=False)]
                for section in content["sections"]
            ],
        }
    article_record = {
        "slug": slug,
        "date": today,
        "category": {language: html.escape(article["category"][language], quote=False) for language in LANGUAGES},
        "sources": [
            {"label": html.escape(source["label"], quote=False), "url": html.escape(source["url"], quote=True)}
            for source in sources
        ],
        "lang": safe_lang,
    }
    updated = dict(existing)
    updated["updated"] = today
    updated["languages"] = list(LANGUAGES)
    updated["articles"] = [article_record] + [
        x for x in known if x.get("slug") != slug
    ]
    page_dir = ROOT / "blog" / slug
    page_file = page_dir / "index.html"
    if page_file.exists():
        raise RuntimeError(f"Refusing to overwrite existing article page: {page_file}")

    if args.dry_run:
        log(f"DRY RUN OK: {article_record['lang']['en']['title']}")
        log(f"Slug: {slug}; sources: {len(sources)}; English words: {sum(len(p.split()) for _, p in article_record['lang']['en']['sections'])}")
        for source in sources:
            log(f"Source: {source['label']} — {source['url']}")
        return 0

    page_dir.mkdir(parents=True, exist_ok=False)
    page_file.write_text(ARTICLE_TEMPLATE.replace("__SLUG__", slug), encoding="utf-8")
    ARTICLES_FILE.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    subprocess.run(["git", "config", "user.name", "github-actions[bot]"], cwd=ROOT, check=True)
    subprocess.run(["git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com"], cwd=ROOT, check=True)
    subprocess.run(["git", "add", "blog/articles.json", str(page_file.relative_to(ROOT))], cwd=ROOT, check=True)
    subprocess.run(["git", "commit", "-m", f"Publish editorial: {slug}"], cwd=ROOT, check=True)
    subprocess.run(["git", "push"], cwd=ROOT, check=True)
    log(f"Published: {slug}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except NoPublish as exc:
        log(f"Publication skipped: {exc}")
        sys.exit(0)
    except Exception as exc:
        log(f"ERROR: {exc}")
        sys.exit(1)
