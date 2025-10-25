from playwright.sync_api import sync_playwright
import json
import requests
from pathlib import Path
import re
 
SEARCH_TERMS = ["deer", "cat", "dog", "snake", "elephant", "flower", "bird", "rabbit", "horse", "fish"]
NUM_PAGES = 3


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)[:50]

def download_file(url, local_filename):
    try:
        with requests.get(url, stream=True, timeout=10) as r:
            r.raise_for_status()
            with open(local_filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"🖼️ Saved {local_filename}")
        return local_filename
    except Exception as e:
        print(f"⚠️ Could not download {url}: {e}")
        return None

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)

    for term in SEARCH_TERMS:
        base_url = f"https://old.reddit.com/r/Dreams/search?q={term}&restrict_sr=on"
        output_file = f"dreams_{term}_posts.json"
        image_dir = Path("images") / term
        image_dir.mkdir(parents=True, exist_ok=True)

        posts = []
        page = browser.new_page()
        page.goto(base_url)

        page_number = 0
        while page_number < NUM_PAGES:
            print(f"📄 Page {page_number + 1}")

            post_links = page.query_selector_all("a.search-title")
            for link_el in post_links:
                title = link_el.inner_text().strip()
                link = link_el.get_attribute("href")
                if link.startswith("/"):
                    link = "https://old.reddit.com" + link

                print(f"➡️ Visiting post: {title}")
                post_page = browser.new_page()
                post_page.goto(link)

        
                body_text = ""
                try:
                    body_el = post_page.query_selector("div.expando div.usertext-body div.md") \
                               or post_page.query_selector("div.entry div.usertext-body div.md")
                    if body_el:
                        body_text = body_el.inner_text().strip()
                except:
                    pass

            
                image_urls = set()

             
                for img in post_page.query_selector_all("img"):
                    src = img.get_attribute("src")
                    if src and src.startswith("http"):
                        image_urls.add(src)

                for a in post_page.query_selector_all("a"):
                    href = a.get_attribute("href")
                    if href and (href.endswith(".jpg") or href.endswith(".png") or href.endswith(".jpeg")):
                        image_urls.add(href)

                local_images = []
                for idx, img_url in enumerate(image_urls):
                    safe_name = sanitize_filename(title) + f"_{idx}.jpg"
                    local_path = image_dir / safe_name
                    if download_file(img_url, local_path):
                        local_images.append(str(local_path))

                if body_text or local_images:
                    posts.append({
                        "title": title,
                        "link": link,
                        "body": body_text,
                        "images": local_images
                    })

                post_page.close()

            next_btn = page.query_selector(".next-button a")
            if not next_btn:
                break
            next_btn.click()
            page.wait_for_timeout(4000)
            page_number += 1

        page.close()

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(posts, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved {len(posts)} posts for '{term}' to {output_file}")
        print(f"✅ Images stored in {image_dir}")

    browser.close()
