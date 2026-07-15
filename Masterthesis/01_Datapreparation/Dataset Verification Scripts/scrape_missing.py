"""Scrapes web content for specific missing or problematic prompts.

This script reads a predefined list of missing prompts, queries the Serper API
for relevant URLs, filters out restricted domains, scrapes the webpage content,
and updates the local file system with new source files and metadata.
"""

import json
import os
import time
import requests
import trafilatura
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get("SERPER_API_KEY")
if not API_KEY:
    raise ValueError("SERPER_API_KEY is missing. Please set it in the .env file.")

OUTPUT_DIR = "scraped_data_full"
MIN_CHARS = 500

BAD_DOMAINS = [
    "youtube.com",
    "youtu.be",
    "linkedin.com",
    "amazon.com",
    "amazon.de",
    "spotify.com",
    "merriam-webster.com",
    "slideshare.net",
    "facebook.com",
    "twitter.com",
    "instagram.com",
    "tiktok.com",
    "quora.com",
    "reddit.com",
]


def search_serper(query, num_results=20):
    """Query the Serper API for web search results based on a text prompt.

    Args:
        query (str): The search query to submit.
        num_results (int, optional): The number of results to request. Defaults to 20.

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


def is_bad_domain(url):
    """Determine whether a given URL belongs to a restricted domain.

    Args:
        url (str): The URL to be evaluated.

    Returns:
        bool: True if the domain is restricted, False otherwise.
    """
    for bd in BAD_DOMAINS:
        if bd in url.lower():
            return True
    return False


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
            return trafilatura.extract(response.text)
        return None
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None


def main():
    """Execute the scraping process for missing prompts.

    This function reads a list of missing prompts from a text file, performs web
    searches for each, scrapes valid URLs avoiding restricted domains, and saves
    the content and metadata locally.
    """
    with open("scratch/rows_to_scrape.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        if not line.strip():
            continue
        prompt_id_str, prompt = line.strip().split("|||")
        prompt_id = int(prompt_id_str)

        prompt_dir = os.path.join(OUTPUT_DIR, f"prompt_{prompt_id:03d}")
        if not os.path.exists(prompt_dir):
            os.makedirs(prompt_dir)

        print(f"\n[{prompt_id}/600] RESCRAPING Prompt: {prompt}")

        search_results = search_serper(prompt, num_results=30)
        if not search_results or "organic" not in search_results:
            print("  No organic results found.")
            continue

        usable_sources = []
        metadata_log = []

        for result in search_results["organic"]:
            if len(usable_sources) >= 5:
                break

            url = result.get("link")
            title = result.get("title")

            if is_bad_domain(url):
                print(f"  -> [SKIPPED] Bad domain: {url}")
                continue

            print(f"  -> Trying URL: {url}")
            content = scrape_url(url)

            if content and len(content) >= MIN_CHARS:
                source_idx = len(usable_sources) + 1
                filename = f"source_{source_idx}.txt"
                filepath = os.path.join(prompt_dir, filename)

                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)

                usable_sources.append(url)
                metadata_log.append(
                    {
                        "source_id": source_idx,
                        "url": url,
                        "title": title,
                        "filename": filename,
                        "character_count": len(content),
                    }
                )
                print(f"     [SUCCESS] Saved as {filename} ({len(content)} chars)")
            else:
                reason = (
                    "Blocked/Failed"
                    if not content
                    else f"Too short ({len(content)} chars)"
                )
                print(f"     [SKIPPED] {reason}")

            time.sleep(1)

        metadata_file = os.path.join(prompt_dir, "metadata.json")
        metadata = {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "sources_collected": len(usable_sources),
            "sources": metadata_log,
        }
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        if len(usable_sources) < 5:
            print(f"  [WARNING] Only found {len(usable_sources)} usable sources.")

        time.sleep(1)

    print("\nRescraping completed.")


if __name__ == "__main__":
    main()
