import requests
from urllib.parse import quote_plus
from datetime import datetime, timedelta, timezone
import feedparser


def fetch_recent_news_for_ticker(ticker: str, days: int = 3, max_items: int = 20):

    # initialize a list of all articles to be collected
    all_articles = []

    # get a string like "AAPL stock" and pass into the google.com/rss/search url
    ticker = quote_plus(f'"{ticker}" stock')
    url = f"https://news.google.com/rss/search?q={ticker}+when:{days}d&hl=en-US&gl=US&ceid=US:en"

    # make a call to the url and look for a response
    response = requests.get(url, timeout=10)
    # raise an error in case of a failure
    response.raise_for_status()
    # parser turns response into a python object easy to iterate over
    news_objects = feedparser.parse(response.text)

    # calculate the oldest acceptable article date
    min_acceptable_article_date = datetime.now(timezone.utc) - timedelta(days=days)

    # iterate over the news articles to collect useful information
    for news in news_objects.entries[:max_items]:

        # if there is a published date, grab it
        published_date = None
        if getattr(news, "published_parsed", None):
            published_date = datetime(*news.published_parsed[:6], tzinfo=timezone.utc)

        # do not include if the article's date is too old
        if published_date and published_date < min_acceptable_article_date:
            continue

        # if there is a publisher, grab it
        publisher = None
        if hasattr(news, "source"):
            if getattr(news, "source", None):
                publisher = getattr(news.source, "title", None)

        # create the results in a dictionary format for each of access
        current_result = {"title": news.get("title"), "link": news.get("link"), "published": published_date.isoformat() if published_date else None, "source": publisher}

        # append the info dictionary to the list of all articles
        all_articles.append(current_result)

    # return a list of all the news articles
    return all_articles
