from firecrawl import FirecrawlApp
from config import settings

class WebScraper:
    def __init__(self):
        if not settings.FIRECRAWL_API_KEY:
            raise ValueError("❌ FIRECRAWL_API_KEY is missing in .env or config.py")
            
        self.app = FirecrawlApp(api_key=settings.FIRECRAWL_API_KEY)

    def scrape_page(self, url: str):
        """
        Uses Firecrawl to turn a URL into clean Markdown.
        """
        print(f"🔥 Firecrawling: {url} ...")
        try:
            # Scrape and get markdown directly
            scrape_result = self.app.scrape_url(url, params={'formats': ['markdown']})
            
            # Extract markdown content
            content = scrape_result.get('markdown', '')
            
            if not content:
                print(f"⚠️ Warning: No content returned for {url}")
                return ""

            print(f"✅ Successfully scraped {len(content)} characters.")
            return f"\n--- SOURCE: {url} ---\n{content}"
            
        except Exception as e:
            print(f"❌ Failed to scrape {url}: {e}")
            return ""

# Test
if __name__ == "__main__":
    try:
        scraper = WebScraper()
        print(scraper.scrape_page("https://example.com"))
    except Exception as e:
        print(e)