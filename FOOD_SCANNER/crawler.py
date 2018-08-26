# Import the libraries
from selenium import webdriver
import json
import os
from urllib.request import urlretrieve
import requests

searchterm = 'pav bhaji' # will also be the name of the folder
url = "https://www.google.co.in/search?q="+searchterm+"&source=lnms&tbm=isch"
# NEED TO DOWNLOAD CHROMEDRIVER, insert path to chromedriver inside parentheses in following line
browser = webdriver.Firefox()
browser.get(url)
header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
counter = 0
succounter = 0

if not os.path.exists(searchterm):
    os.mkdir(searchterm)

for _ in range(1000):
    browser.execute_script("window.scrollBy(0,10000)")
    
image_links = []
image_types = []

for x in browser.find_elements_by_xpath('//div[contains(@class,"rg_meta")]'):
    counter = counter + 1
    try:
        print("URL:",json.loads(x.get_attribute('innerHTML'))["ou"])
        image_links.append(json.loads(x.get_attribute('innerHTML'))["ou"])
        image_types.append(json.loads(x.get_attribute('innerHTML'))['ity'])
    except:
            print ("can't get img")
            
browser.close()

for i in range(0, len(image_links)):
    try: 
        r = requests.get(image_links[i])
        extension = "jpg" if image_types[i] == "" else image_types[i]
        with open(searchterm + "/" + searchterm + "_" + str(i) + "." + extension, 'wb') as outfile:
            outfile.write(r.content)
        print(i+1, "/", len(image_links), "done!")
    except:
        continue
    
