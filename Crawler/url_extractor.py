"""Fetching and link extraction for English Wikipedia articles.

`URLExtractor` is the second seam: given an article URL it returns the article
URLs linked from that page. `WikipediaURLExtractor` owns the network call, so
the rate limiter lives here — that is the only place a request is issued.
"""

import time
from abc import ABC, abstractmethod
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup

from rate_limiter import RateLimiter

WIKI_HOST = "en.wikipedia.org"
WIKI_PATH_PREFIX = "/wiki/"

# Wikipedia answers the default python-requests user agent with HTTP 403, so a
# descriptive one is required.
USER_AGENT = (
    "interviews-crawler/1.0 (educational exercise; "
    "https://github.com/harshwardhan6062/interviews)"
)

REQUEST_TIMEOUT_SECONDS = 10.0
POLL_INTERVAL_SECONDS = 0.05


class FetchError(Exception):
    """A page could not be retrieved (network failure, timeout, HTTP error)."""


class URLExtractor(ABC):
    """Returns the article URLs linked from a given article."""

    @abstractmethod
    def extract(self, wikipedia_url: str) -> list[str]:
        """Fetch `wikipedia_url` and return the article URLs it links to.

        Every returned URL is absolute and normalized, so callers can use them
        directly as set keys. An empty list means the page genuinely had no
        qualifying links; a failed fetch raises `FetchError` instead, so the
        two cases stay distinguishable.
        """
        raise NotImplementedError


class WikipediaURLExtractor(URLExtractor):
    """Extracts `https://en.wikipedia.org/wiki/<page_name>` links over HTTP."""

    def __init__(self, rate_limiter: RateLimiter) -> None:
        self._rate_limiter = rate_limiter

    def extract(self, wikipedia_url: str) -> list[str]:
        """See `URLExtractor.extract`."""
        html = self._fetch_page(wikipedia_url)
        return self._parse_links(wikipedia_url, html)

    def _fetch_page(self, url: str) -> str:
        """Block until the rate limiter permits a request, then GET `url`.

        This is the only network call in the program, which is why the limiter
        is consulted here. `allow()` is boolean-only, so waiting means polling.
        """
        while not self._rate_limiter.allow():
            time.sleep(POLL_INTERVAL_SECONDS)

        try:
            response = requests.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise FetchError(f"{type(exc).__name__}: {exc}") from exc

        return response.text

    def _parse_links(self, page_url: str, html: str) -> list[str]:
        """Pull normalized article links out of `html`. Pure, no network."""
        soup = BeautifulSoup(html, "html.parser")

        links: list[str] = []
        seen: set[str] = set()
        for anchor in soup.find_all("a", href=True):
            url = self.normalize(page_url, anchor["href"])
            if url is not None and url not in seen:
                seen.add(url)
                links.append(url)

        return links

    @staticmethod
    def normalize(page_url: str, href: str) -> str | None:
        """Canonicalize `href` found on `page_url`, or None if not crawlable.

        Returns None for anything that is not an English Wikipedia article:
        other hosts and languages, non-HTTP schemes, `/w/index.php` style
        paths, and namespaced pages (`File:`, `Category:`, `Special:`, ...).
        """
        try:
            parts = urlsplit(urljoin(page_url, href.strip()))
        except ValueError:
            return None

        if parts.scheme not in ("http", "https"):
            return None
        if parts.netloc != WIKI_HOST:
            return None
        if not parts.path.startswith(WIKI_PATH_PREFIX):
            return None

        title = parts.path[len(WIKI_PATH_PREFIX):]
        if not title:
            return None
        # A colon means a non-article namespace. Checked after unquoting so
        # percent-encoded forms like File%3AFoo.jpg are caught too. This also
        # drops the rare real article whose title contains a colon.
        if ":" in unquote(title):
            return None

        # Drop the fragment and query: .../Google#History and
        # .../Google?action=edit are the same page as .../Google. Doing this
        # before the caller dedupes is what stops one article being re-crawled.
        return urlunsplit(("https", WIKI_HOST, parts.path, "", ""))
