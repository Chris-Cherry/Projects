
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
import csv
import time
import random

def get_calls():
    """Use html links found with get_lightning_round_htmls and find out
    the bull and bear calls for each together with the date."""

    # Read htmls we found earlier and step through them
    html_csv = open('htmls.csv', 'r')
    html_reader = csv.reader(html_csv)
    htmls = []
    for row in html_reader:
        htmls.append(row[0])
    html_csv.close()

    try:
        fin_csv = open('fin.csv', 'r')
        fin_reader = csv.reader(fin_csv)
        fins = []
        for row in fin_reader:
            fins.append(row[0])
        fin_csv.close()
    except:
        fins = []

    # Don't rerun old HTMLs
    html_set = set(htmls)
    fin_set = set(fins)

    html_leftover = html_set - fin_set
    htmls = list(html_leftover)

    # Subset the htmls before changing tor IP
    n_htmls = 10
    htmls = htmls[0:n_htmls]

    # Finally append htmls to fin as we go
    fin_csv = open('fin.csv', 'a')
    fin_writer = csv.writer(fin_csv)

    ticker_csv = open('CSR.csv', 'a')
    ticker_writer = csv.writer(ticker_csv)

    if len(htmls) is 0:
        print('No more HTMLs')
        return

    profile = webdriver.FirefoxProfile()
    profile.set_preference('network.proxy.type', 1)
    profile.set_preference('network.proxy.http', 'localhost')
    profile.set_preference('network.proxy.http_port', 8123)
    profile.set_preference('network.proxy.ssl', 'localhost')
    profile.set_preference('network.proxy.ssl_port', 8123)
    profile.set_preference('network.proxy.ftp', 'localhost')
    profile.set_preference('network.proxy.ftp_port', 8123)
    driver = webdriver.Firefox(profile)

    for row in htmls:
        print(row)
        driver.get(row)

        try:
            # Get the author of the article. There are two major formats,
            # a different one for each author plus on sub author.
            auth_element = driver.find_element_by_xpath("//*[@data-nick]")
            auth = auth_element.get_attribute('data-slug')

            # Get the date the stock was recommended. Articles are
            # published 1 day after the show.
            time_element = driver.find_element_by_xpath("//time[@itemprop='datePublished']")
            time_ticker = time_element.get_attribute('content')

        except NoSuchElementException:
            continue

        # Search strategy for articles
        print('miriam-metzinger' not in auth)
        if 'miriam-metzinger' not in auth:
            # Get bullish stock paragraph elements between bull and
            # bear headers
            bull_elements = driver.find_elements_by_xpath(
                "//p[preceding-sibling::h3='Bullish Calls' and following-sibling::h3='Bearish Calls']")
            # Get bull tickers
            bull_tickers = []
            for bull_element in bull_elements:
                try:
                    ticker_element = bull_element.find_element_by_xpath("./a[@class='ticker-link']")
                    ticker = ticker_element.get_attribute("symbol")
                    bull_tickers.append(ticker)
                except NoSuchElementException:
                    continue

            # Do the same process for bearish stock elements
            bear_elements = driver.find_elements_by_xpath("//p[preceding-sibling::h3='Bearish Calls']")
            # Last three elements are not stock paragraph elements
            bear_tickers = []
            for bear_element in bear_elements:
                try:
                    ticker_element = bear_element.find_element_by_xpath("./a[@class='ticker-link']")
                    ticker = ticker_element.get_attribute("symbol")
                    bear_tickers.append(ticker)
                except NoSuchElementException:
                    continue

        else:
            # Get block quote elements. The first is bull the second is bear.
            try:
                bull_quote = driver.find_element_by_xpath("//blockquote[1]")
                bull_elements = bull_quote.find_elements_by_xpath("./p")
                bull_tickers = []
                for bull_element in bull_elements:
                    try:
                        ticker_element = bull_element.find_element_by_xpath(".//a[@class='ticker-link']")
                        ticker = ticker_element.get_attribute("symbol")
                        bull_tickers.append(ticker)
                    except NoSuchElementException:
                        continue
            except NoSuchElementException:
                pass

            try:
                bear_quote = driver.find_element_by_xpath("//blockquote[2]")
                bear_elements = bear_quote.find_elements_by_xpath("./p")
                bear_tickers = []
                for bear_element in bear_elements:
                    try:
                        ticker_element = bear_element.find_element_by_xpath(".//a[@class='ticker-link']")
                        ticker = ticker_element.get_attribute("symbol")
                        bear_tickers.append(ticker)
                    except NoSuchElementException:
                        continue
            except NoSuchElementException:
                pass

        # Finally ticker symbols need to be written to the csv file.
        if 'bull_tickers' in locals():
            if len(bull_tickers) > 0:
                for bull_ticker in bull_tickers:
                    ticker_writer.writerow([bull_ticker, 'Bull', time_ticker])

        if 'bear_tickers' in locals():
            if len(bear_tickers) > 0:
                for bear_ticker in bear_tickers:
                    ticker_writer.writerow([bear_ticker, 'Bear', time_ticker])

        fin_writer.writerow([row])

    driver.quit()
    fin_csv.close()
    ticker_csv.close()



def main():
    get_calls()

if __name__ == '__main__':
    main()


