#!/usr/bin/env python
# coding: utf-8

# In[1]:


import urllib.request as ul
from bs4 import BeautifulSoup
import requests
import re
from datetime import datetime as dt
import pandas as pd


# In[17]:


senatorwiki = 'https://en.wikipedia.org/wiki/List_of_current_United_States_Senators'
html = requests.get(senatorwiki)
soup = BeautifulSoup(html.text)
senatortable = soup.find('table',{'class':"sortable"})
rows = senatortable.findAll('tr')

state = []
name = []
party = []
age = []
occupation = []
prev = []
assoff = []
termup = []
res = []

for tr in rows:
    cells=tr.findAll('td')
    alt = tr.findAll('th')
    if len(cells)==9:
        #state.append(cells[0].find(text=True))
        party.append(cells[2].find(text=True))
        occupation.append(cells[4].find(text=True))
        prev.append(cells[5].find(text=True))
        assoff.append(cells[6].find(text=True))
        termup.append(cells[7].find(text=True))
        res.append(cells[8].find(text=True))
        age.append(int(cells[3].find('span','noprint ForceAgeToShow').string[6:-1]))
    if len(cells)==10:
        state.append(cells[0].find(text=True))
        party.append(cells[3].find(text=True))
        occupation.append(cells[5].find(text=True))
        prev.append(cells[6].find(text=True))
        assoff.append(cells[7].find(text=True))
        termup.append(cells[8].find(text=True))
        res.append(cells[9].find(text=True))
        age.append(int(cells[4].find('span','noprint ForceAgeToShow').string[6:-1]))
    if len(alt)==1:
        name.append(alt[0].find(text=True))

c=rows[0].findAll('th')
stac = [el.find(text=True) for el in c]
cats = [el.rstrip() for el in stac]
cats.remove('Image')
cats[3] = 'Age'
zcat = dict(zip([str(x) for x in range(len(cats))],cats))

assoff = [x.rstrip() for x in assoff]
assoff = [dt.strptime(x,'%B %d, %Y') for x in assoff]
occupation = [x.rstrip() for x in occupation]
termup = [x.rstrip() for x in termup]
termup = [dt.strptime(x,'%Y') for x in termup]
states = [x for x in state for i in range(2)]
prev = [x.rstrip() for x in prev]

senators = pd.DataFrame([states,name,party,age,occupation,prev,assoff,termup,res],index=cats).T[]


# In[21]:


senators

