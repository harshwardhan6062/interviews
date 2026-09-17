"""Breadth-first crawler for English Wikipedia articles.

Usage:
    python crawler.py https://en.wikipedia.org/wiki/Google 3 3
                       ^start_page                         ^max  ^requests/min
"""

import argparse
import sys
import time
from collections import deque

from rate_limiter import SlidingWindowRateLimiter
from url_extractor import FetchError, URLExtractor, WikipediaURLExtractor


class Crawler:
    """Crawls outward from `start_url`, breadth-first, visiting each page once."""

    def __init__(
        self,
        start_url: str,
        max_num_pages: int,
        url_extractor: URLExtractor,
    ) -> None:
        if max_num_pages < 1:
            raise ValueError(f"max_num_pages must be >= 1, got {max_num_pages}")

        self.start_url = start_url
        self.max_num_pages = max_num_pages
        self.url_extractor = url_extractor

        self.queue: deque[str] = deque([start_url])
        self.seen: set[str] = {start_url}
        self.visited: list[str] = []

    def crawl(self) -> list[str]:
        """Crawl until the page budget is spent or the frontier empties.

        A FIFO queue makes the traversal breadth-first: every link on the start
        page is crawled before any link found on those pages.
        """
        while self.queue and len(self.visited) < self.max_num_pages:
            url = self.queue.popleft()

            try:
                links = self.url_extractor.extract(url)
            except FetchError as exc:
                # The request was still made, so it counted against the rate
                # limit, but an unreachable page is not a visited page.
                print(f"skip  {url}: {exc}", file=sys.stderr)
                continue

            self.visited.append(url)
            print(f"[{len(self.visited)}/{self.max_num_pages}] {url}", file=sys.stderr)

            if len(self.visited) >= self.max_num_pages:
                break

            self._enqueue(links)

        return self.visited

    def _enqueue(self, links: list[str]) -> None:
        """Queue any link not seen before.

        Membership is checked before insert, so a URL is queued at most once in
        the crawl's lifetime and can never be fetched twice.
        """
        for link in links:
            if link not in self.seen:
                self.seen.add(link)
                self.queue.append(link)


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="crawler.py",
        description="Breadth-first crawl of English Wikipedia articles.",
    )
    parser.add_argument(
        "start_page",
        help="article to start from, e.g. https://en.wikipedia.org/wiki/Google",
    )
    parser.add_argument(
        "max_num_pages",
        type=_positive_int,
        help="stop after visiting this many pages",
    )
    parser.add_argument(
        "rate_limit",
        type=_positive_int,
        help="maximum requests per minute",
    )
    args = parser.parse_args(argv)

    # Validate the start page with the same rule applied to every other URL,
    # rather than a second, slightly different check.
    start_url = WikipediaURLExtractor.normalize(args.start_page, args.start_page)
    if start_url is None:
        parser.error(f"not an English Wikipedia article URL: {args.start_page}")

    rate_limiter = SlidingWindowRateLimiter(args.rate_limit)
    url_extractor = WikipediaURLExtractor(rate_limiter)
    crawler = Crawler(start_url, args.max_num_pages, url_extractor)

    started = time.monotonic()
    try:
        visited = crawler.crawl()
    except KeyboardInterrupt:
        # A low rate limit means long waits, so Ctrl-C is expected rather than
        # exceptional. Report what was crawled instead of dumping a traceback.
        visited = crawler.visited
        print("\ninterrupted", file=sys.stderr)
    elapsed = time.monotonic() - started

    for position, url in enumerate(visited, start=1):
        print(f"{position}. {url}")
    print(f"visited {len(visited)} page(s) in {elapsed:.1f}s", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
