import time
import sys

try:
    # Latest SDK uses 'Firecrawl' class
    from firecrawl import Firecrawl
except ImportError:
    # Fallback/Safety: If user has old SDK, try to alias it, 
    # but strongly recommend updating via 'pip install --upgrade firecrawl-py'
    try:
        from firecrawl import FirecrawlApp as Firecrawl
    except ImportError:
        print("❌ Critical Error: 'firecrawl-py' not found. Please run: pip install firecrawl-py")
        sys.exit(1)

from config import settings

class WebScraper:
    def __init__(self):
        if not settings.FIRECRAWL_API_KEY:
            raise ValueError("❌ FIRECRAWL_API_KEY is missing in .env or config.py")
            
        self.app = Firecrawl(api_key=settings.FIRECRAWL_API_KEY)

    def scrape_page(self, url: str):
        """
        Uses Firecrawl to crawl the website and extract markdown.
        
        Using the LATEST 'crawl' method (v1 SDK).
        """
        print(f"🔥 Firecrawling site: {url} ...")
        
        try:
            # Fix: Pass arguments directly instead of wrapping in 'params' dict.
            # Use 'scrape_options' (snake_case) for the Python SDK.
            crawl_status = self.app.crawl(
                url, 
                limit=20,
                scrape_options={'formats': ['markdown']},
                poll_interval=2
            )
            
            # Validate response
            if not crawl_status or 'data' not in crawl_status:
                print(f"⚠️ Warning: No content returned for {url}")
                print(f"Debug Info: {crawl_status}")
                return ""

            combined_content = ""
            page_count = 0
            
            # The data structure contains a list of pages in 'data'
            for item in crawl_status['data']:
                # Handle cases where metadata might be missing
                metadata = item.get('metadata', {})
                source_url = metadata.get('sourceURL', url) if metadata else url
                markdown = item.get('markdown', '')
                
                if markdown:
                    combined_content += f"\n\n--- SOURCE: {source_url} ---\n{markdown}"
                    page_count += 1
            
            print(f"✅ Successfully crawled {page_count} pages.")
            return combined_content
            
        except AttributeError:
            print("❌ SDK Mismatch: It seems you are using an older version of 'firecrawl-py'.")
            print("   Please run: pip install --upgrade firecrawl-py")
            return ""
        except TypeError as e:
            print(f"❌ Argument Error: {e}")
            return ""
        except Exception as e:
            print(f"❌ Failed to crawl {url}: {e}")
            return ""

# Test
if __name__ == "__main__":
    try:
        scraper = WebScraper()
        # Test with the documentation site
        print(scraper.scrape_page("https://docs.firecrawl.dev"))
    except Exception as e:
        print(e)