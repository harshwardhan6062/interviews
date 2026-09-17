"""Shortest link path between two English Wikipedia articles.

Fetching is parallelised across a thread pool; the rate limit stays global
because every worker shares one lock-guarded `RateLimiter`.

Usage:
    python3 crawler_2.py search https://en.wikipedia.org/wiki/Google \
                                https://en.wikipedia.org/wiki/Online_advertising
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

from rate_limiter import SlidingWindowRateLimiter
from url_extractor import FetchError, URLExtractor, WikipediaURLExtractor

DEFAULT_REQUESTS_PER_MINUTE = 60
DEFAULT_MAX_WORKERS = 8

# Safety bound. Wikipedia articles carry hundreds of links each, so an
# unreachable target would otherwise mean an unbounded crawl: hop 2 alone is
# already hundreds of thousands of pages. Raise it to search harder.
DEFAULT_MAX_NUM_PAGES = 200


class Search:
    """Finds a link path from `start_url` to `end_url`, breadth-first.

    BFS explores in order of increasing hop count, so the first path found is a
    shortest one. `parents` maps each discovered URL to the page it was
    discovered on; walking those links back from `end_url` rebuilds the path.

    The frontier is expanded one hop level at a time: every page in a level is
    handed to the thread pool at once, then results are consumed in submission
    order. That keeps two properties that out-of-order completion would break —
    the path found is still a shortest one, and the output is identical to a
    sequential run, because parents are assigned in the same order either way.
    """

    def __init__(
        self,
        start_url: str,
        end_url: str,
        url_extractor: URLExtractor,
        max_num_pages: int = DEFAULT_MAX_NUM_PAGES,
        max_workers: int = DEFAULT_MAX_WORKERS,
    ) -> None:
        if max_num_pages < 1:
            raise ValueError(f"max_num_pages must be >= 1, got {max_num_pages}")
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")

        self.start_url = start_url
        self.end_url = end_url
        self.url_extractor = url_extractor
        self.max_num_pages = max_num_pages
        self.max_workers = max_workers

        # Doubles as the visited set: a key being present means the URL has
        # already been discovered, so it is never queued twice. The start page
        # maps to None, which is what terminates the walk back.
        self.parents: dict[str, str | None] = {start_url: None}
        self.pages_fetched = 0

    def search(self) -> list[str] | None:
        """Return the path from start to end inclusive, or None if not found.

        Only the worker threads touch the network; `parents` and the frontier
        are read and written by this thread alone, so they need no locking.
        The one piece of genuinely shared state is the rate limiter, which
        guards itself.
        """
        if self.start_url == self.end_url:
            return [self.start_url]

        pool = ThreadPoolExecutor(max_workers=self.max_workers)
        try:
            level = [self.start_url]
            while level and self.pages_fetched < self.max_num_pages:
                # Never submit more fetches than the page budget allows.
                budget = self.max_num_pages - self.pages_fetched
                batch = level[:budget]

                # Workers block inside extract() until the shared limiter
                # grants them a token, so submitting the whole level at once
                # cannot outrun the rate limit.
                futures = [
                    pool.submit(self.url_extractor.extract, url) for url in batch
                ]

                next_level: list[str] = []
                for url, future in zip(batch, futures):
                    try:
                        links = future.result()
                    except FetchError as exc:
                        # The request was still made, so it counted against the
                        # rate limit, but an unreachable page is not a visit.
                        print(f"skip  {url}: {exc}", file=sys.stderr)
                        continue

                    path = self._consume(url, links, next_level)
                    if path is not None:
                        return path

                level = next_level
        finally:
            # Don't wait on in-flight fetches; the answer is already known.
            pool.shutdown(wait=False, cancel_futures=True)

        return None

    def _consume(
        self,
        url: str,
        links: list[str],
        next_level: list[str],
    ) -> list[str] | None:
        """Record parents for one page's links; return a path if end was found."""
        self.pages_fetched += 1
        print(f"[{self.pages_fetched}/{self.max_num_pages}] {url}", file=sys.stderr)

        for link in links:
            if link in self.parents:
                continue
            self.parents[link] = url

            # The target only has to be *seen* as a link, not fetched, so this
            # returns one request earlier than crawling it would.
            if link == self.end_url:
                return self._build_path()

            next_level.append(link)

        return None

    def _build_path(self) -> list[str]:
        """Walk `parents` back from `end_url` and reverse into start -> end."""
        path: list[str] = []
        node: str | None = self.end_url
        while node is not None:
            path.append(node)
            node = self.parents[node]
        path.reverse()
        return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="crawler_2.py",
        description="Find a link path between two English Wikipedia articles.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    search_parser = subparsers.add_parser(
        "search",
        help="breadth-first search for a shortest link path",
    )
    search_parser.add_argument(
        "start_page",
        help="article to start from, e.g. https://en.wikipedia.org/wiki/Google",
    )
    search_parser.add_argument(
        "end_page",
        help="article to reach",
    )
    search_parser.add_argument(
        "--rate-limit",
        type=int,
        default=DEFAULT_REQUESTS_PER_MINUTE,
        help=f"maximum requests per minute (default {DEFAULT_REQUESTS_PER_MINUTE})",
    )
    search_parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"parallel fetches (default {DEFAULT_MAX_WORKERS})",
    )
    args = parser.parse_args(argv)

    # Validate both endpoints with the same rule applied to every crawled URL.
    start_url = WikipediaURLExtractor.normalize(args.start_page, args.start_page)
    if start_url is None:
        parser.error(f"not an English Wikipedia article URL: {args.start_page}")
    end_url = WikipediaURLExtractor.normalize(args.end_page, args.end_page)
    if end_url is None:
        parser.error(f"not an English Wikipedia article URL: {args.end_page}")

    rate_limiter = SlidingWindowRateLimiter(args.rate_limit)
    url_extractor = WikipediaURLExtractor(rate_limiter)
    search = Search(start_url, end_url, url_extractor, max_workers=args.workers)

    try:
        path = search.search()
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 1

    if path is None:
        print(
            f"no path found after fetching {search.pages_fetched} page(s)",
            file=sys.stderr,
        )
        return 1

    for url in path:
        print(url)
    print(
        f"found a {len(path) - 1}-hop path after fetching "
        f"{search.pages_fetched} page(s)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
