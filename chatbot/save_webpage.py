from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urlparse, urljoin
import shutil
import os
import logging

logger = logging.getLogger(__name__)

def setup_webdriver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    return webdriver.Chrome(options=chrome_options)

def save_webpage(url, filename):
    driver = setup_webdriver()
    
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        
        logger.info(f"Saved {filename}")
        
        if not os.path.exists('data'):
            os.makedirs('data')
        shutil.move(filename, os.path.join('data', filename))
        
        logger.info(f"Moved {filename} to data folder")
        
    finally:
        driver.quit()

def get_blog_links(main_url):
    driver = setup_webdriver()
    
    try:
        driver.get(main_url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        main_domain = urlparse(main_url).netloc
        
        blog_links = [
            urljoin(main_url, a['href'])
            for a in soup.find_all('a', href=True)
            if urlparse(urljoin(main_url, a['href'])).netloc == main_domain
        ]
    
    finally:
        driver.quit()
    
    return blog_links

def main():
    with open('links.txt', 'r') as file:
        blog_urls = [line.strip() for line in file if line.strip()]
    
    for i, blog_url in enumerate(blog_urls):
        blog_links = get_blog_links(blog_url)
        
        for j, link in enumerate(blog_links):
            filename = f"file_{i+1}_{j+1}.html"
            save_webpage(link, filename)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
