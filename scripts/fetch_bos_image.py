# fetch_bos_images.py  (fetch_bos_docs.py 실행 후 돌릴 것)
import re, time, pathlib
from urllib.parse import urlparse, urljoin, quote, unquote
import requests

OUT = pathlib.Path("bos_docs")
ASSETS = OUT / "_assets"
BASE = "https://developers.bos-semi.com"

session = requests.Session()
session.headers["User-Agent"] = "bos-docs-mirror/1.0 (research)"


def get(url, tries=3):
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


ASSETS.mkdir(parents=True, exist_ok=True)
seen = {}

for md_file in OUT.rglob("*.md"):
    if md_file.name == "INDEX.md":
        continue
    text = md_file.read_text(encoding="utf-8")
    src = re.search(r"^source:\s*(\S+)", text, re.M)
    if not src:
        continue

    page = get(src.group(1))
    if not page:
        continue

    imgs = re.findall(r'<img[^>]+src="([^"]+)"', page.text)
    imgs += re.findall(r'<(?:source|image)[^>]+(?:src|href)="([^"]+\.(?:svg|png|jpe?g))"',
                       page.text, re.I)

    links = []
    for u in set(imgs):
        if u.startswith("data:"):
            continue
        full = urljoin(BASE, u)
        name = pathlib.Path(unquote(urlparse(full).path)).name
        dst = ASSETS / name

        if name not in seen:
            r = get(full)
            if not r:
                continue
            dst.write_bytes(r.content)
            seen[name] = full
            print("saved", name)
            time.sleep(0.3)

        rel = pathlib.os.path.relpath(dst, md_file.parent).replace("\\", "/")
        links.append(f"![{name}]({rel})  <!-- {full} -->")

    if links:
        md_file.write_text(text.rstrip() + "\n\n## 이미지\n\n" + "\n".join(links) + "\n",
                           encoding="utf-8")

print(f"\n완료: {len(seen)}개 이미지 → {ASSETS}/")