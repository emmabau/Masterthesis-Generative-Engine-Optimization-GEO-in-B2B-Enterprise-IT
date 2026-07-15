"""Automates the replacement of problematic or insufficient web sources.

This script identifies scraped data sources from excluded domains or those that
are too short, queries the Serper API for better alternatives, scrapes the new
content, and updates the local dataset metadata and files accordingly.
"""

import json
import os
import time
import requests
import trafilatura
from urllib.parse import urlparse

API_KEY = "4160e6317db4645d5395d8fc2bd6aea5477c413a"
OUTPUT_DIR = "scraped_data_full"
BAD_DOMAINS = [
    "google",
    "facebook",
    "twitter",
    "instagram",
    "tiktok",
    "quora",
    "linkedin",
    "amazon",
    "spotify",
    "merriam-webster",
    "slideshare",
    "youtube",
    "youtu.be",
]
MIN_CHARS = 500


def is_bad_domain(url):
    """Determine whether a given URL belongs to a restricted domain.

    Args:
        url (str): The URL to be evaluated.

    Returns:
        bool: True if the domain is restricted or invalid, False otherwise.
    """
    try:
        domain = urlparse(url).netloc.lower()
        return any(bad in domain for bad in BAD_DOMAINS)
    except Exception:
        return True


def search_serper(query, num_results=50):
    """Query the Serper API for web search results based on a text prompt.

    Args:
        query (str): The search query to submit.
        num_results (int, optional): The number of results to request. Defaults to 50.

    Returns:
        dict: A dictionary containing search results parsed from JSON, or None if
        the request fails.
    """
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query, "num": num_results})
    headers = {"X-API-KEY": API_KEY, "Content-Type": "application/json"}
    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error calling Serper API for query '{query}': {e}")
        return None


def scrape_url(url):
    """Retrieve and extract main text content from a specified URL.

    Args:
        url (str): The URL of the webpage to scrape.

    Returns:
        str: The extracted text content, or None if the request or extraction fails.
    """
    try:
        response = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            },
        )
        if response.status_code == 200:
            downloaded = response.text
            text = trafilatura.extract(downloaded)
            return text
        return None
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None


def main():
    """Execute the source replacement process for the dataset.

    This function iterates through all scraped prompt directories, identifies sources
    that are too short or from restricted domains, and replaces them by querying new
    sources and updating local files.
    """
    prompts_to_fix = []

    for i in range(1, 601):
        prompt_dir = os.path.join(OUTPUT_DIR, f"prompt_{i:03d}")
        metadata_file = os.path.join(prompt_dir, "metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    bad_indices = []
                    existing_urls = []
                    for idx, source in enumerate(data.get("sources", [])):
                        url = source.get("url", "").lower()
                        char_count = source.get("character_count", 0)
                        existing_urls.append(url)

                        if char_count < MIN_CHARS or is_bad_domain(url):
                            bad_indices.append(idx)

                    if bad_indices:
                        prompts_to_fix.append(
                            {
                                "prompt_id": i,
                                "prompt_dir": prompt_dir,
                                "metadata_file": metadata_file,
                                "prompt_text": data.get("prompt", ""),
                                "bad_indices": bad_indices,
                                "existing_urls": existing_urls,
                                "data": data,
                            }
                        )
                except Exception as e:
                    print(f"Error reading {metadata_file}: {e}")

    print(f"Found {len(prompts_to_fix)} prompts containing short or bad links.")

    for item in prompts_to_fix:
        prompt_id = item["prompt_id"]
        prompt_text = item["prompt_text"]
        print(f"\n[{prompt_id}] Fixing Prompt: {prompt_text}")

        search_results = search_serper(prompt_text, num_results=50)
        if not search_results or "organic" not in search_results:
            print("  No organic results found.")
            continue

        organic_results = search_results["organic"]
        used_urls = set(item["existing_urls"])
        data = item["data"]
        sources = data["sources"]

        for b_idx in item["bad_indices"]:
            target_source = sources[b_idx]
            old_url = target_source["url"]
            old_chars = target_source.get("character_count", 0)
            source_id = target_source["source_id"]
            filename = target_source["filename"]
            print(f"  Replacing [{source_id}] (Chars: {old_chars}) {old_url}")

            replacement_found = False
            for result in organic_results:
                new_url = result.get("link")
                new_title = result.get("title")

                new_url_lower = new_url.lower()
                if new_url_lower in used_urls or is_bad_domain(new_url_lower):
                    continue

                print(f"  -> Trying URL: {new_url}")
                content = scrape_url(new_url)

                if content and len(content) >= MIN_CHARS:
                    filepath = os.path.join(item["prompt_dir"], filename)

                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(content)

                    target_source["url"] = new_url
                    target_source["title"] = new_title
                    target_source["character_count"] = len(content)

                    used_urls.add(new_url_lower)
                    replacement_found = True
                    print(
                        f"     [SUCCESS] Replaced with {filename} ({len(content)} chars)"
                    )
                    break
                else:
                    reason = (
                        "Blocked/Failed"
                        if not content
                        else f"Too short ({len(content)} chars)"
                    )
                    print(f"     [SKIPPED] {reason}")

                time.sleep(1)

            if not replacement_found:
                print(
                    f"  [WARNING] Could not find a replacement for source {source_id}"
                )

        with open(item["metadata_file"], "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        time.sleep(1)

    print("\nReplacement process completed.")


if __name__ == "__main__":
    main()
