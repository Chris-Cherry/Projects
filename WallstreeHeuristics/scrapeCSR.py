from selenium import webdriver
import csv
import time
import random

def get_lightning_round_htmls():
    """Scrape article html links for all lightning round articles from
    seekingAlpha"""

    html_csv = open('htmls.csv', 'w')
    html_writer = csv.writer(html_csv)

    driver = webdriver.Firefox()
    page = 1
    while page <= 1:
        time.sleep(random.uniform(2, 5))
        try:
            url = 'https://seekingalpha.com/stock-ideas/cramers-picks?page='+str(page)
            driver.get(url)
            article_titles = driver.find_elements_by_class_name('a-title')
            article_htmls = list(map(lambda x: x.get_attribute('href'), article_titles))
            lightning_htmls = list(filter(lambda x: 'lightning-round' in x, article_htmls))
            for x in lightning_htmls:
                html_writer.writerow([x])
            page += 1

        except Exception as e:
            print(e)
    driver.close()
    html_csv.close()


def get_calls():
    """Use html links found with get_lightning_round_htmls and find out
    the bull and bear calls for each together with the date."""

    html_csv = open('htmls.csv', 'r')
    html_reader = csv.reader(html_csv)

    driver = webdriver.Firefox()
    for row in html_reader:
        print(row[0])
        driver.get(row[0])

def main():
    #get_lightning_round_htmls()
    get_calls()

if __name__ == '__main__':
    main()


