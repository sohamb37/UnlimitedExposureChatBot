from firecrawl import Firecrawl
import json

# Initialize the Firecrawl App with your API key
# If self-hosting, you would point this to your local instance
app = Firecrawl(api_key="fc-07a6b3de972c42f28a29be9a098bc10e")

url = "https://www.rustcheck.ca/"

print(f"Starting crawl for: {url}")

# 2. Use the new 'crawl' method
# Note: In the new SDK, arguments like 'limit' are passed directly
crawl_result = app.crawl(
    url,
    limit=10,
    scrape_options={
        'formats': ['markdown'] 
    }
)

# 3. Save to disk (The structure of the result is slightly different now)
# The data is usually inside 'data' or the object itself is iterable depending on the specific sub-version.
# We convert it to a dict/list for safe saving.

try:
    # If the result is an object, we try to access its 'data' attribute
    data_to_save = crawl_result.get('data', []) if isinstance(crawl_result, dict) else crawl_result.data
except AttributeError:
    # Fallback if the structure is raw
    data_to_save = crawl_result

with open("/home/paritosh/hybrid_bot/data/scraped_data.json", "w", encoding="utf-8") as f:
    # default=str helps serialize objects that JSON doesn't know how to handle
    json.dump(data_to_save, f, indent=4, ensure_ascii=False, default=str)

print("Saved data to scraped_data.json")