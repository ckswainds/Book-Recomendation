import requests
import json
import pandas as pd
import time

def fetch_papers(query, limit=100, max_results=300):
    """
    Fetch research papers from Semantic Scholar API.
    query: search keyword (e.g., "machine learning")
    limit: results per API call (max 100)
    max_results: total number of results to fetch
    """
    papers = []
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    fields = "title,abstract,authors,url,year,citationCount,venue"

    for offset in range(0, max_results, limit):
        url = f"{base_url}?query={query}&limit={limit}&offset={offset}&fields={fields}"
        response = requests.get(url)

        if response.status_code != 200:
            print("Error fetching", query, ":", response.status_code)
            break

        data = response.json()
        items = data.get("data", [])
        if not items:
            break

        # Attach query label for reference
        for item in items:
            item["searchQuery"] = query
        papers.extend(items)

        time.sleep(1)  # avoid rate limit

    return papers


def save_papers(papers, json_file="papers.json", csv_file="papers.csv"):
    """
    Save papers to JSON (raw) and CSV (refined).
    """
    # Save raw JSON
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=4, ensure_ascii=False)

    # Refine for CSV
    refined = []
    for p in papers:
        refined.append({
            "SearchQuery": p.get("searchQuery", ""),
            "Title": p.get("title", ""),
            "Abstract": p.get("abstract", ""),
            "Authors": ", ".join([a.get("name", "") for a in p.get("authors", [])]),
            "Year": p.get("year", ""),
            "Citations": p.get("citationCount", 0),
            "Venue": p.get("venue", ""),
            "URL": p.get("url", "")
        })

    df = pd.DataFrame(refined)
    df.to_csv(csv_file, index=False, encoding="utf-8")
    print(f"Saved {len(refined)} papers → {json_file}, {csv_file}")


if __name__ == "__main__":
    queries = [
        "machine learning",
        "deep learning",
        "artificial intelligence",
        "natural language processing",
        "nlp",
        "computer vision",
        "reinforcement learning",
        "data science"
    ]

    all_papers = []
    for q in queries:
        print(f"Fetching papers for: {q}")
        papers = fetch_papers(q, limit=100, max_results=300)  # up to 300 per query
        all_papers.extend(papers)

    save_papers(all_papers, json_file="all_papers.json", csv_file="all_papers.csv")
