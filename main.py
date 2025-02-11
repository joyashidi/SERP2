import requests
import concurrent.futures
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from googlesearch import search
import spacy
from collections import Counter

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Define the search query
query = "Crime reporting papers site:researchgate.net OR site:sciencedirect.com OR site:springer.com"

# Function to fetch search results
def get_search_results(query, num_results=10):
    results = []
    for url in search(query, num_results=num_results):
        results.append(url)
    return results

# Function to scrape text content from a webpage
def scrape_content(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract meta description if full text isn't available
        meta_desc = soup.find("meta", attrs={"name": "description"})
        title = soup.title.string if soup.title else ""

        if meta_desc:
            text = meta_desc["content"]
        else:
            paragraphs = soup.find_all("p")
            text = " ".join([p.get_text() for p in paragraphs])

        return title + " " + text  # Combine title with scraped text
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

# Function to extract keywords using NLP
def extract_features(text):
    doc = nlp(text.lower())
    keywords = [token.lemma_ for token in doc if token.is_alpha and token.pos_ in ["NOUN", "ADJ"]]
    return keywords

# Multithreaded function to scrape and process content
def process_urls(urls):
    feature_counts = Counter()
    url_features = {}  # Dictionary to store keywords per URL

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(scrape_content, url): url for url in urls}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            text = future.result()
            if text:
                features = extract_features(text)
                feature_counts.update(features)
                url_features[url] = features[:5]  # Store top 5 keywords per URL

    return feature_counts, url_features

# Main function
def main():
    urls = get_search_results(query, num_results=10)
    print(f"Found {len(urls)} search results.\n")

    feature_counts, url_features = process_urls(urls)

    # Print URLs and their extracted keywords
    print("Top keywords from each site:\n")
    for url, keywords in url_features.items():
        print(f"{url}\n → {', '.join(keywords)}\n")

    # Get the top 10 most common features
    top_features = feature_counts.most_common(10)
    features, counts = zip(*top_features)

    # Visualization
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(features), y=list(counts), hue=list(features), legend=False, palette="viridis")
    plt.xticks(rotation=45)
    plt.xlabel("Feature")
    plt.ylabel("Count")
    plt.title("Top 10 Distinctive Features in Crime Reporting Papers")
    plt.show()

if __name__ == "__main__":
    main()
