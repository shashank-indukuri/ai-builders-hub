import asyncio
from crawl4ai import AsyncWebCrawler


async def main():
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://www.nbcnews.com/sports",
        )
        print(result.markdown)

if __name__ == "__main__":
    asyncio.run(main())