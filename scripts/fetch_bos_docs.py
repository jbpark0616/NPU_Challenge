# fetch_bos_docs.py
import gzip, io, re, time, pathlib
from urllib.parse import urlparse, urljoin, quote, unquote
from xml.etree import ElementTree

import requests, trafilatura

SITEMAP = "https://developers.bos-semi.com/sitemap.xml"
CANON_HOST = "developers.bos-semi.com"     # 도메인 정규화 대상
OUT = pathlib.Path("bos_docs")
DELAY = 0.5                                 # 서버 예의상 딜레이(초)
NS = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}

session = requests.Session()
session.headers["User-Agent"] = "bos-docs-mirror/1.0 (research)"


def get(url, tries=3):
    """공백 포함 URL 인코딩 + 재시도."""
    p = urlparse(url)
    safe = p._replace(path=quote(unquote(p.path))).geturl()
    for i in range(tries):
        try:
            r = session.get(safe, timeout=30)
            r.raise_for_status()
            return r
        except Exception as e:
            if i == tries - 1:
                print(f"  [FAIL] {url} -> {e}")
                return None
            time.sleep(2 * (i + 1))


def parse_sitemap(url, seen=None):
    """sitemap index(중첩)와 .gz를 모두 처리하며 URL 전부 수집."""
    seen = seen or set()
    if url in seen:
        return []
    seen.add(url)

    r = get(url)
    if not r:
        return []

    raw = r.content
    if raw[:2] == b"\x1f\x8b":                      # gzip 매직 넘버
        raw = gzip.decompress(raw)

    root = ElementTree.fromstring(raw)
    tag = root.tag.split("}")[-1]

    if tag == "sitemapindex":                        # 중첩 sitemap
        out = []
        for loc in root.findall(".//s:loc", NS):
            out += parse_sitemap(loc.text.strip(), seen)
        return out

    return [loc.text.strip() for loc in root.findall(".//s:loc", NS)]


def normalize(url):
    return urlparse(url)._replace(netloc=CANON_HOST, scheme="https").geturl()


def out_path(url):
    """URL 경로를 디렉터리 구조로 보존."""
    path = unquote(urlparse(url).path).strip("/")
    path = re.sub(r"^docs/?", "", path) or "index"
    parts = [re.sub(r"[^\w\-.]+", "_", p) for p in path.split("/")]
    return OUT.joinpath(*parts).with_suffix(".md")


def main():
    OUT.mkdir(exist_ok=True)
    urls = [normalize(u) for u in parse_sitemap(SITEMAP)]
    urls = sorted(set(u for u in urls if "/docs/" in u))
    print(f"sitemap에서 {len(urls)}개 문서 페이지 발견\n")

    index = []
    for i, url in enumerate(urls, 1):
        r = get(url)
        if not r:
            continue

        md = trafilatura.extract(
            r.text, output_format="markdown",
            include_tables=True, include_links=True, favor_recall=True,
        )
        if not md or len(md.strip()) < 50:
            print(f"  [SKIP] 본문 없음: {url}")
            continue

        m = re.search(r"<title>(.*?)</title>", r.text, re.S)
        title = m.group(1).split("|")[0].strip() if m else "Untitled"

        dst = out_path(url)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(
            f"---\ntitle: {title}\nsource: {url}\n---\n\n{md}\n",
            encoding="utf-8",
        )
        index.append((title, dst.relative_to(OUT).as_posix(), url))
        print(f"[{i}/{len(urls)}] {dst.relative_to(OUT)}")
        time.sleep(DELAY)

    lines = ["# BOS SDK 문서 미러\n", f"총 {len(index)}개 페이지\n"]
    lines += [f"- [{t}]({p}) — {u}" for t, p, u in sorted(index)]
    (OUT / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n완료: {len(index)}개 저장 → {OUT}/  (목록: {OUT}/INDEX.md)")


if __name__ == "__main__":
    main()