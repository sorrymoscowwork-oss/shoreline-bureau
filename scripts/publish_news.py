#!/usr/bin/env python3
"""SHORELINE BUREAU editorial publisher.

Uses public RSS research and the free-tier Gemini GenerateContent API, validates a multilingual
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
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from datetime import datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, quote, urlencode
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
MODEL = "gemini-3.5-flash-lite"
API_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"


def log(message: str) -> None:
    print(f"[SHORELINE] {message}", flush=True)


def gemini_request(prompt: str) -> dict:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key: raise RuntimeError("GitHub secret GEMINI_API_KEY is not available to this workflow.")
    payload = {"model": MODEL, "input": prompt, "tools": [{"type": "google_search"}]}
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "x-goog-api-key": key}
    transient = {429, 500, 502, 503, 504}
    retry_delays = (30, 60, 120, 180)
    for attempt in range(len(retry_delays) + 1):
        req = Request(API_URL, data=body, method="POST", headers=headers)
        try:
            with urlopen(req, timeout=600) as response:
                data = json.loads(response.read().decode("utf-8"))
            if data.get("status") != "completed":
                raise RuntimeError("Gemini interaction did not complete: " + json.dumps({"status": data.get("status"), "error": data.get("error")}, ensure_ascii=False)[:1200])
            return data
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1600]
            if exc.code in transient and attempt < len(retry_delays):
                delay = retry_delays[attempt]
                log(f"Gemini Interactions API HTTP {exc.code}; retrying in {delay}s.")
                time.sleep(delay)
                continue
            raise RuntimeError(f"Gemini Interactions API HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            if attempt < len(retry_delays):
                delay = retry_delays[attempt]
                log(f"Gemini connection error; retrying in {delay}s.")
                time.sleep(delay)
                continue
            raise RuntimeError("Could not reach Gemini API: " + str(exc.reason)) from exc
    raise RuntimeError("Gemini interaction failed after retries.")


def create_interaction(prompt: str) -> dict:
    return gemini_request(prompt)


def extract_output(response: dict) -> tuple[str, list[dict]]:
    chunks, citations = [], []
    for step in response.get("steps", []):
        if step.get("type") != "model_output": continue
        for block in step.get("content", []):
            if block.get("type") != "text": continue
            if block.get("text"): chunks.append(block["text"])
            for annotation in block.get("annotations", []):
                if annotation.get("type") == "url_citation":
                    url = str(annotation.get("url") or "").strip()
                    title = str(annotation.get("title") or "").strip()
                    if url.startswith("https://"): citations.append({"url": url, "title": title or url})
    if not chunks and response.get("output_text"): chunks.append(response["output_text"])
    return "\n".join(chunks).strip(), citations


def candidate_finish_reason(response: dict) -> str:
    return str(response.get("status", "UNKNOWN"))


def parse_json(text: str) -> dict:
    # Accept plain JSON or a fenced JSON block; parse the first complete object
    # rather than slicing to the final brace (which may be inside trailing text).
    cleaned = re.sub(r"^\\s*```(?:json)?\\s*|\\s*```\\s*$", "", text.strip(), flags=re.I)
    start = cleaned.find("{")
    if start < 0:
        raise RuntimeError("Gemini did not return a JSON article.")
    try:
        value, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    except json.JSONDecodeError as exc:
        raise RuntimeError("Gemini returned invalid article JSON: " + str(exc)) from exc
    if not isinstance(value, dict):
        raise RuntimeError("Gemini response JSON must be an object.")
    return value


def article_quality_error(article: dict) -> str | None:
    if article.get("no_publish"):
        return None
    if not isinstance(article.get("category"), dict) or not isinstance(article.get("lang"), dict):
        return "Missing category or language data."
    for language in LANGUAGES:
        content = article["lang"].get(language)
        if not article["category"].get(language) or not isinstance(content, dict):
            return f"Missing {language} category or article."
        if not content.get("title") or not content.get("excerpt"):
            return f"Missing {language} title or excerpt."
        sections = content.get("sections")
        if not isinstance(sections, list) or not 5 <= len(sections) <= 7:
            return f"{language} article must contain 5–7 sections."
        if any(not isinstance(section, list) or len(section) != 2 or
               not all(str(part).strip() for part in section) for section in sections):
            return f"Malformed section in {language} article."
    english_words = sum(
        len(re.findall(r"\\b[\\w’'-]+\\b", paragraph))
        for _, paragraph in article["lang"]["en"]["sections"]
    )
    if english_words < 800:
        return f"English article is too short ({english_words} words; minimum 800)."
    return None


def generate_article(prompt: str) -> dict:
    # Keep the complete original brief in context for the repair attempt. A repair
    # must correct both syntax and content completeness, not merely produce parseable JSON.
    previous_output = ""
    last_error = ""
    for attempt in range(2):
        request_prompt = prompt
        if attempt == 1:
            request_prompt = (
                prompt
                + "\\n\\nREPAIR REQUIREMENT: Your previous response failed validation. "
                + last_error
                + " Return a complete replacement JSON object, not a patch. "
                + "The English article MUST contain at least 800 words in its section paragraphs; "
                + "write 850–1,100 words. Each of en, ru, de, zh MUST contain 5–7 complete "
                + "sections with full translated article text. Do not shorten translations. "
                + "Preserve source IDs and factual caveats. Return only valid JSON, with no fences. "
                + "\\n\\nPREVIOUS RESPONSE (for correction only):\\n"
                + previous_output[:24000]
            )
        response = create_interaction(request_prompt)
        output, citations = extract_output(response)
        previous_output = output
        finish_reason = candidate_finish_reason(response)
        if not output:
            last_error = f"Gemini returned no text (finishReason={finish_reason})."
        else:
            try:
                article = parse_json(output)
                last_error = article_quality_error(article) or ""
                if not last_error:
                    article["_grounding_sources"] = citations
                    return article
            except RuntimeError as exc:
                last_error = str(exc)
        if attempt == 0:
            log(f"{last_error} (finishReason={finish_reason}) Requesting one complete repair attempt.")
        else:
            raise RuntimeError(f"Gemini article failed after one repair attempt: {last_error} (finishReason={finish_reason})")
    raise RuntimeError("Gemini article generation failed.")


def fetch_article_page(url: str) -> dict | None:
    """Fetch readable public article text using only Python standard library."""
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; SHORELINE-BUREAU-NewsBot/1.0)"})
    try:
        with urlopen(request, timeout=8) as response:
            final_url = response.geturl()
            if "html" not in response.headers.get("Content-Type", "").lower():
                return None
            raw = response.read(1200000).decode("utf-8", errors="replace")
        title_match = re.search(r"<title[^>]*>(.*?)</title>", raw, re.I | re.S)
        title = html.unescape(re.sub(r"<[^>]+>", " ", title_match.group(1))).strip() if title_match else ""
        desc_match = re.search(r'<meta[^>]+(?:name|property)=["\'](?:description|og:description)["\'][^>]+content=["\'](.*?)["\']', raw, re.I | re.S)
        description = html.unescape(desc_match.group(1))[:600] if desc_match else ""
        # Remove non-editorial blocks before extracting visible text.
        raw = re.sub(r"(?is)<(script|style|nav|footer|header|aside|noscript).*?</\1>", " ", raw)
        paragraphs = re.findall(r"(?is)<p[^>]*>(.*?)</p>", raw)
        text = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", p)) for p in paragraphs)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 450:
            text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw))).strip()
        if len(text) < 450:
            return None
        date_match = re.search(r'"datePublished"\s*:\s*"([^"]+)"', raw, re.I)
        if not date_match:
            date_match = re.search(r"""<time[^>]+datetime=["']([^"']+)""", raw, re.I)
        published = date_match.group(1) if date_match else ""
        return {"url": final_url, "title": title, "description": description, "text": text[:6500], "published": published}
    except Exception as exc:
        log(f"Could not read source page {url[:100]}: {exc}")
        return None


def fetch_news_items() -> list[dict]:
    """Collect recent Red Sea news from publisher RSS/Atom feeds and a local-news sitemap."""
    feeds = [
        ("Egypt Independent", "https://www.egyptindependent.com/feed/"),
        ("Daily News Egypt", "https://dailynewsegypt.com/feed/"),
        ("Egyptian Streets", "https://egyptianstreets.com/feed/"),
        ("Al-Masry Al-Youm", "https://www.almasryalyoum.com/rss/rss"),
        ("Al-Shorouk", "https://www.shorouknews.com/rss.aspx"),
        ("Cairo 24", "https://www.cairo24.com/rss"),
        ("Sada El Balad", "https://www.elbalad.news/rss"),
    ]
    cutoff = datetime.now(ZoneInfo("UTC")).timestamp() - 30 * 86400
    collected, seen = [], set()
    # Strict geographic scope: Hurghada and its nearby resort/town areas only.
    # Do not include distant Red Sea Governorate locations or other countries.
    relevant = (
        "hurghada", "الغردقة", "el gouna", "الجونة",
        "sahl hasheesh", "سهل حشيش", "makadi bay", "makadi", "مكادي",
        "soma bay", "سوما باي", "safaga", "سفاجا",
        "el ahyaa", "الاحياء", "al ahyaa", "old vic", "old sherry",
        "الغردقة والمناطق المحيطة", "مدينة الغردقة"
    )

    def local_name(tag: str) -> str:
        return tag.rsplit("}", 1)[-1].lower()

    def add_candidate(headline: str, link: str, pub_date: str, publisher: str,
                      summary_html: str, feed_name: str) -> bool:
        if not headline or not link:
            return False
        summary = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", summary_html))).strip()
        published_at = None
        if pub_date:
            try:
                if re.fullmatch(r"\\d{8}T\\d{6}Z", pub_date):
                    published_at = datetime.strptime(pub_date, "%Y%m%dT%H%M%SZ").replace(tzinfo=ZoneInfo("UTC")).timestamp()
                else:
                    published_at = parsedate_to_datetime(pub_date).timestamp()
            except Exception:
                try:
                    published_at = datetime.fromisoformat(pub_date.replace("Z", "+00:00")).timestamp()
                except Exception:
                    published_at = None
        page = fetch_article_page(link)
        page_text = page["text"] if page else ""
        if published_at is None and page and page.get("published"):
            try:
                published_at = datetime.fromisoformat(page["published"].replace("Z", "+00:00")).timestamp()
            except Exception:
                try:
                    published_at = parsedate_to_datetime(page["published"]).timestamp()
                except Exception:
                    published_at = None
        if published_at is None or published_at < cutoff:
            return False
        searchable = (headline + " " + summary + " " + page_text[:2500]).lower()
        if not any(term in searchable for term in relevant):
            return False
        article_text = page_text if len(page_text) >= 450 else summary
        if len(article_text) < 250:
            return False
        final_url = page["url"] if page else link
        parsed = urlparse(final_url)
        if parsed.scheme != "https" or not parsed.netloc:
            return False
        domain = parsed.netloc.lower().removeprefix("www.")
        canonical = f"https://{domain}{parsed.path}".rstrip("/")
        if canonical in seen:
            return False
        seen.add(canonical)
        collected.append({
            "id": f"N{len(collected)+1}",
            "title": (page["title"] if page and page["title"] else headline)[:300],
            "publisher": publisher or feed_name or domain,
            "domain": domain,
            "published": pub_date,
            "url": final_url,
            "description": (page["description"] if page else summary[:600]),
            "article_text": article_text[:6500],
        })
        return True

    for feed_name, feed_url in feeds:
        feed_collected = 0
        request = Request(feed_url, headers={"User-Agent": "Mozilla/5.0 (compatible; SHORELINE-BUREAU/1.0)", "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml"})
        try:
            with urlopen(request, timeout=15) as response:
                root = ET.fromstring(response.read())
            nodes = [node for node in root.iter() if local_name(node.tag) in ("item", "entry")]
            for node in nodes:
                fields = {local_name(child.tag): child for child in list(node)}
                headline = (fields.get("title").text or "").strip() if fields.get("title") is not None else ""
                link_node = fields.get("link")
                link = ""
                if link_node is not None:
                    link = (link_node.attrib.get("href") or link_node.text or "").strip()
                date_node = fields.get("pubdate") or fields.get("published") or fields.get("updated") or fields.get("date")
                pub_date = (date_node.text or "").strip() if date_node is not None else ""
                desc_node = fields.get("encoded") or fields.get("description") or fields.get("summary") or fields.get("content")
                summary = (desc_node.text or "") if desc_node is not None else ""
                source_node = fields.get("source")
                publisher = (source_node.text or "").strip() if source_node is not None else feed_name
                if add_candidate(headline, link, pub_date, publisher, summary, feed_name):
                    feed_collected += 1
                if feed_collected >= 15:
                    break
            log(f"RSS {feed_name}: {feed_collected} regional candidate(s); {len(collected)} total.")
        except Exception as exc:
            log(f"RSS {feed_name}: unavailable or invalid feed ({type(exc).__name__}: {str(exc)[:180]}).")

    # GDELT DOC 2.0 is a free global news index. Search broadly across publishers,
    # including pages surfaced from public social platforms, but anchor every query
    # to Hurghada and its immediate neighbouring areas in Egypt.
    # One consolidated request avoids hammering the public GDELT endpoint.
    # It searches only the approved Hurghada-area locations over the last 30 days.
    query = '("Hurghada" OR "الغردقة" OR "El Gouna" OR "الجونة" OR "Sahl Hasheesh" OR "سهل حشيش" OR "Makadi Bay" OR "Makadi" OR "مكادي" OR "Soma Bay" OR "سوما باي" OR "Safaga" OR "سفاجا" OR "El Ahyaa" OR "الاحياء")'
    now_utc = datetime.now(ZoneInfo("UTC"))
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": "100",
        "sort": "datedesc",
        "startdatetime": (now_utc - timedelta(days=30)).strftime("%Y%m%d") + "000000",
        "enddatetime": now_utc.strftime("%Y%m%d") + "235959",
    }
    endpoint = "https://api.gdeltproject.org/api/v2/doc/doc?" + urlencode(params)
    request = Request(endpoint, headers={"User-Agent": "Mozilla/5.0 (compatible; SHORELINE-BUREAU/1.0)"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=35) as response:
                raw_payload = response.read().decode("utf-8", errors="replace")
            payload = json.loads(raw_payload)
            articles = payload.get("articles", []) if isinstance(payload, dict) else []
            found = 0
            for item in articles:
                headline = str(item.get("title") or "").strip()
                link = str(item.get("url") or "").strip()
                date_value = str(item.get("seendate") or "").strip()
                domain = str(item.get("domain") or "").strip()
                if add_candidate(headline, link, date_value, domain, headline, "GDELT"):
                    found += 1
            log(f"GDELT consolidated search: {len(articles)} indexed result(s), {found} usable Hurghada-area candidate(s).")
            break
        except HTTPError as exc:
            if exc.code == 429 and attempt < 2:
                delay = 20 * (attempt + 1)
                log(f"GDELT rate limited (429); waiting {delay}s before retry {attempt + 2}/3.")
                time.sleep(delay)
                continue
            log(f"GDELT search unavailable (HTTP {exc.code}); continuing with other public sources.")
            break
        except (URLError, json.JSONDecodeError, TimeoutError) as exc:
            if attempt < 2:
                delay = 8 * (attempt + 1)
                log(f"GDELT temporary error ({type(exc).__name__}); retrying in {delay}s.")
                time.sleep(delay)
                continue
            log(f"GDELT search unavailable ({type(exc).__name__}: {str(exc)[:180]}); continuing with other public sources.")
            break
        except Exception as exc:
            log(f"GDELT search unavailable ({type(exc).__name__}: {str(exc)[:180]}); continuing with other public sources.")
            break

    # Local Hurghada publisher feeds (WordPress-style RSS endpoints).
    for feed_name, feed_url in [
        ("Hurghada News", "https://hurghada.news/feed/"),
        ("Hurghada News — local news", "https://hurghada.news/category/news/feed/"),
        ("Hurghada News — business", "https://hurghada.news/category/business/feed/"),
        ("Hurghada News — environment", "https://hurghada.news/category/environment/feed/"),
        ("Hurghada News — real estate", "https://hurghada.news/category/real-estate/feed/"),
    ]:
        feed_collected = 0
        request = Request(feed_url, headers={"User-Agent": "Mozilla/5.0 (compatible; SHORELINE-BUREAU/1.0)", "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml"})
        try:
            with urlopen(request, timeout=15) as response:
                root = ET.fromstring(response.read())
            nodes = [node for node in root.iter() if local_name(node.tag) in ("item", "entry")]
            for node in nodes:
                fields = {local_name(child.tag): child for child in list(node)}
                headline_node = fields.get("title")
                headline = (headline_node.text or "").strip() if headline_node is not None else ""
                link_node = fields.get("link")
                link = ""
                if link_node is not None:
                    link = (link_node.attrib.get("href") or link_node.text or "").strip()
                date_node = fields.get("pubdate") or fields.get("published") or fields.get("updated") or fields.get("date")
                pub_date = (date_node.text or "").strip() if date_node is not None else ""
                desc_node = fields.get("encoded") or fields.get("description") or fields.get("summary") or fields.get("content")
                summary = (desc_node.text or "") if desc_node is not None else ""
                if add_candidate(headline, link, pub_date, feed_name, summary, feed_name):
                    feed_collected += 1
                if feed_collected >= 15:
                    break
            log(f"RSS {feed_name}: {feed_collected} regional candidate(s); {len(collected)} total.")
        except Exception as exc:
            log(f"RSS {feed_name}: unavailable or invalid feed ({type(exc).__name__}: {str(exc)[:180]}).")

    # Hurghada News is a live regional publisher with dated article pages.
    # Discover links from its real category pages; article URLs use /article/<slug>.
    local_pages = [
        ("Hurghada News — News", "https://hurghada.news/category/news"),
        ("Hurghada News — Business", "https://hurghada.news/category/business"),
        ("Hurghada News — Travel", "https://hurghada.news/category/travel"),
        ("Hurghada News — Real Estate", "https://hurghada.news/category/real-estate"),
        ("Hurghada News — Environment", "https://hurghada.news/category/environment"),
        ("Hurghada News — Community", "https://hurghada.news/category/community"),
    ]
    for feed_name, page_url in local_pages:
        found = 0
        request = Request(page_url, headers={"User-Agent": "Mozilla/5.0 (compatible; SHORELINE-BUREAU/1.0)"})
        try:
            with urlopen(request, timeout=15) as response:
                listing_html = response.read(1200000).decode("utf-8", errors="replace")
            links = {}
            for match in re.finditer(r"""<a[^>]+href=["'](https://hurghada\.news/article/[^"']+)["'][^>]*>(.*?)</a>""", listing_html, re.I | re.S):
                url = html.unescape(match.group(1))
                headline = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", match.group(2)))).strip()
                if headline and url not in links:
                    links[url] = headline
            log(f"Local listing {feed_name}: discovered {len(links)} article link(s).")
            for url, headline in list(links.items())[:20]:
                page = fetch_article_page(url)
                if not page:
                    continue
                if add_candidate(headline or page["title"], page["url"], page.get("published", ""),
                                 "Hurghada News", page.get("description", "") + " " + page["text"],
                                 feed_name):
                    found += 1
            log(f"Local listing {feed_name}: {found} recent regional candidate(s); {len(collected)} total.")
        except Exception as exc:
            log(f"Local listing {feed_name}: unavailable ({type(exc).__name__}: {str(exc)[:180]}).")

    selected, domains = [], set()
    for item in collected:
        if item["domain"] not in domains:
            selected.append(item)
            domains.add(item["domain"])
    for item in collected:
        if item not in selected and len(selected) < 25:
            selected.append(item)
    log(f"Regional discovery complete: {len(collected)} unique candidate(s), {len(selected)} sent to Gemini.")
    return selected[:25]


def article_prompt(existing: dict, today: str) -> str:
    archive = [{"slug": x.get("slug"), "date": x.get("date"), "sources": [z.get("url") for z in x.get("sources", [])]} for x in existing.get("articles", [])[:40]]
    return f"""
You are SHORELINE BUREAU independent editorial desk in Hurghada, Egypt. Date: {today}.

USE GOOGLE SEARCH NOW. Research the open web broadly for concrete, verifiable news published
or updated within the last 30 days. Search multiple queries in English and Arabic, and other
languages where useful. Do not rely on memory. Search across official announcements, news
outlets, tourism and hotel industry, property developers, company statements, public community
pages and social posts indexed publicly. Public Telegram/Facebook material may be used only
when accessible without login or bypassing restrictions. Do not limit discovery to RSS or fixed portals.

STRICT GEOGRAPHY: Hurghada and its districts (including El Ahyaa), El Gouna, Sahl Hasheesh,
Makadi Bay, Soma Bay and Safaga, Egypt. Exclude Marsa Alam, El Quseir, Port Ghalib, Ras Ghareb,
Shalateen, Halayeb, other distant Red Sea locations and every other country. National Egyptian
developments qualify only when sources establish a direct, specific effect on the approved area.

Select one significant local development from the last 30 days. Relevant beats: tourism, hotels,
airport and flights, real estate, investment, infrastructure, roads, utilities, healthcare, education,
ecology, culture, local prices, residency and expat life. One credible substantive source is enough;
corroboration is welcome. Attribute claims; distinguish announcements, plans, works and completions.
Never invent facts, dates, quotes, figures, approvals, prices or timelines. Skip if evidence is weak.
Do not repeat the archive: {json.dumps(archive, ensure_ascii=False)}

Write 850–1,100 English words in 5–7 sections, including what happened, location, actors, verified
status/timeline, unknowns and local significance. End with an independent SHORELINE BUREAU
assessment. Neutral, analytical, non-promotional.

Produce complete full versions in English, Russian, German and Simplified Chinese. Preserve all facts,
numbers, caveats and assessment consistently.

Return only JSON with this schema:
{{"no_publish":false,"reason":"","slug":"ascii-kebab-case","category":{{"en":"...","ru":"...","de":"...","zh":"..."}},
"sources":[{{"title":"...","url":"https://..."}}],
"lang":{{"en":{{"title":"...","excerpt":"...","sections":[["Heading","Paragraph"]]}},
"ru":{{"title":"...","excerpt":"...","sections":[["Заголовок","Текст"]]}},
"de":{{"title":"...","excerpt":"...","sections":[["Überschrift","Text"]]}},
"zh":{{"title":"...","excerpt":"...","sections":[["标题","正文"]]}}}}}}
5–7 sections per language; final section is SHORELINE BUREAU assessment. Source URLs must be
actual URLs returned/cited by Google Search; do not invent them. Return no_publish=true if no
credible local story or no verifiable source.
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
    if len(sources) < 1:
        raise RuntimeError("No credible publisher source was selected; refusing to publish.")
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
    if english_words < 800:
        raise RuntimeError(f"English article is too short ({english_words} words; minimum 800).")


class NoPublish(Exception):
    pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Generate and validate, but do not commit.")
    args = parser.parse_args()

    today = datetime.now(ZoneInfo("Africa/Cairo")).date().isoformat()
    existing = json.loads(ARTICLES_FILE.read_text(encoding="utf-8"))
    log(f"Starting editorial research for {today}; dry_run={args.dry_run}")
    article = generate_article(article_prompt(existing, today))
    if article.get("no_publish"):
        log(f"Skipping publication: {article.get('reason', 'No verified local story found.')}")
        return 0
    sources = collect_sources(article.get("_grounding_sources", []))
    if not sources:
        raise RuntimeError("Google Search returned no verifiable grounding citations; refusing to publish.")
    log(f"Gemini Google Search supplied {len(sources)} verifiable source citation(s).")
    slug = clean_slug(article.get("slug", ""), "hurghada-update")
    known = existing.get("articles", [])
    known_slugs = {x.get("slug") for x in known}
    if slug in known_slugs: slug = f"{slug}-{today}"
    source_urls = {source["url"] for source in sources}
    for old in known:
        old_urls = {source.get("url") for source in old.get("sources", [])}
        if source_urls & old_urls:
            raise RuntimeError(f"Duplicate source story detected; overlaps existing article {old.get('slug')}.")
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
