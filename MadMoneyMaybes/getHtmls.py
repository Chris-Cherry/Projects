from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
import csv
import time
import random


def get_lightning_round_htmls():
    """Scrape article html links for all lightning round articles from
    seekingAlpha"""

    html_csv = open('htmls.csv', 'w')
    html_writer = csv.writer(html_csv)
    profile = webdriver.FirefoxProfile()
    profile.set_preference('network.proxy.type', 1)
    profile.set_preference('network.proxy.http', 'localhost')
    profile.set_preference('network.proxy.http_port', 8123)
    profile.set_preference('network.proxy.ssl', 'localhost')
    profile.set_preference('network.proxy.ssl_port', 8123)
    profile.set_preference('network.proxy.ftp', 'localhost')
    profile.set_preference('network.proxy.ftp_port', 8123)

    driver = webdriver.Firefox(profile)

    # How many pages of Mad Money articles are we sorting through?
    # 88 max as of 8/6/18
    page = 1
    while page <= 88:
        # Don't query the site too often
        try:
            # Build a url
            url = 'https://seekingalpha.com/stock-ideas/cramers-picks?page=' + \
                  str(page)
            driver.get(url)
            time.sleep(random.uniform(2, 5))
            # Pull out article title elements, get the href, and select
            # lightning round articles.
            article_titles = driver.find_elements_by_class_name('a-title')
            article_htmls = list(map(lambda x: x.get_attribute('href'),
                                     article_titles))
            lightning_htmls = list(filter(lambda x: 'lightning-round' in x,
                                          article_htmls))
            # Write the htmls to our csv file.
            for x in lightning_htmls:
                html_writer.writerow([x])
            page += 1

        except Exception as e:
            print(e)
    driver.quit()
    html_csv.close()


def main():
    get_lightning_round_htmls()


if __name__ == '__main__':
    main()