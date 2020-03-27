
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import requests
from bs4 import BeautifulSoup
link = "https://www.holidify.com/explore/"
P_link = requests.get(link)
print(P_link)

P_html = P_link.text
P_soup = BeautifulSoup(P_html, "html.parser")
containers = P_soup.findAll("div", {"class" : "col-12 col-md-6 pr-md-3 result"})
print(len(containers))
container = containers[0]
column = ['Place','State', 'Ratings', 'About','Price', 'Attraction']
Places = pd.DataFrame(columns = column)
for container in containers:
    p_name = container.findAll("h2", {"class":"card-heading"})
    p_nameN = p_name[0].text[4:].strip().split()
    if len(p_nameN) == 2:        
          p_nameP = p_nameN[0]
          p_nameP = p_nameP.replace(',','')
          p_nameC = p_nameN[1]
    elif len(p_nameN) == 3:
          p_nameP = p_nameN[0]
          p_nameP = p_nameP.replace(',','')
          p_nameC = p_nameN[1] + " " + p_nameN[2]
    elif len(p_nameN) == 4:
          p_nameP = p_nameN[0]
          p_nameP = p_nameP.replace(',','')
          p_nameC = p_nameN[1] + " " + p_nameN[2] + " " + p_nameN[3]      
    else:
          p_nameP = p_nameN[0]
          p_nameC = "NaN"
    p_rating = container.findAll("span", {"class" : "rating-badge"})
    p_rating = p_rating[0].text[1:4]
    p_about = container.findAll("p",{"class": "card-text"})
    p_about = p_about[0].text
    p_price = container.findAll("p",{"class": "collection-cta"})
    if len(p_price) == 1:
        p_num = p_price[0].text.replace(',','')
        p_numb = re.findall(r'\d+', p_num)
        num = ""
        for i in p_numb:
            num += i
    else:
        num = "NaN"
    p_attraction = container.findAll("div", {"class":"content-card-footer"})
    p_attraction = p_attraction[0].text[:-12].strip()
    
    Data = pd.DataFrame([[p_nameP ,p_nameC, p_rating , p_about, num , p_attraction]])
    Data.columns = column
    Places = Places.append(Data, ignore_index = True)
print(Places.head())
Places.to_csv("Places.csv", index = None)