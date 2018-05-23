
# coding: utf-8

# In[5]:

"""
@author Rahul Gupta
"""

import string
import re
import tweepy
import json
import twitterClient
from collections import Counter
import networkx as nx

import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import opinion_lexicon

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy.stats.stats import pearsonr

from colorama import Fore, Style
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic("config InlineBackend.figure_format = 'retina'")
plt.style.use('seaborn')
pd.set_option('display.max_columns', None)  
pd.options.display.max_colwidth = 200


# In[6]:

import nltk
nltk.download('stopwords')


# In[7]:

from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
nltk.download('opinion_lexicon')


# In[8]:

## Extension 1.1: Reddit Sentiment Analysis


# In[9]:

"""
@author Megha Mohan
"""
import praw
import time
import random
import sys
import csv
import math
import numpy as np
import pandas as pd
from collections import Counter
import networkx as nx

import nltk
nltk.download('punkt')

from colorama import Fore, Style

import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz
from IPython.display import display
from IPython.display import clear_output
from pprint import pprint
from ipywidgets import IntSlider, Output
from IPython.display import clear_output
import datetime as dt
import matplotlib.pyplot as plt
get_ipython().magic("config InlineBackend.figure_format = 'retina'")
plt.style.use('seaborn')
pd.set_option('display.max_columns', None)  
pd.options.display.max_colwidth = 200
import seaborn as sns


# In[10]:

reddit = praw.Reddit(client_id='PiURtsGqnsmLEA',
client_secret='JodQ8pfG4OJ1TiYQPvkjcCAlG6s',
user_agent='SocialMedia')


# In[11]:

headlines = set()
for submission in reddit.subreddit('streaming').new(limit=None):
    headlines.add(submission.title)
clear_output()


# In[12]:

from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

sia = SIA()
results = []

for line in headlines:
    pol_score = sia.polarity_scores(line)
    pol_score['headlines'] = line
    results.append(pol_score)

pprint(results[:3], width=100)


# In[13]:

df = pd.DataFrame.from_records(results)
df.head()


# In[14]:

df['label'] = 0
df.loc[df['compound'] > 0.2, 'label'] = 1
df.loc[df['compound'] < -0.2, 'label'] = -1
df.head()


# In[15]:

df2 = df[['headlines','label']]


# In[16]:

df2.to_csv('reddit1_headlines_labels.csv', mode='a', encoding='utf-8', index=False)


# In[17]:

df.label.value_counts()


# In[18]:

print("Positive headlines:\n")
pprint(list(df[df['label'] == 1].headlines)[:5], width=200)

print("\nNegative headlines:\n")
pprint(list(df[df['label'] == -1].headlines)[:5], width=200)


# In[19]:

df.label.value_counts(normalize=True) * 100


# In[20]:

fig, ax = plt.subplots(figsize=(8, 8))
counts = df.label.value_counts(normalize=True) * 100
sns.barplot(x=counts.index, y=counts, ax=ax)
ax.set_xticklabels(['Postive', 'Neutral', 'Negative'])
ax.set_ylabel("Percentage")
plt.show()


# In[21]:

from nltk.tokenize import word_tokenize, RegexpTokenizer


# In[22]:

example = "Thor is the best movie ever"
print(word_tokenize(example, language='english'))


# In[23]:

tokenizer = RegexpTokenizer(r'\w+')
print(tokenizer.tokenize(example))


# In[24]:

from nltk.corpus import stopwords

stop_words = stopwords.words('english')
print(stop_words[:20])


# In[25]:

def process_text(headlines):
    tokens = []
    for line in headlines:
        line = line.lower()
        toks = tokenizer.tokenize(line)
        toks = [t for t in toks if t not in stop_words]
        tokens.extend(toks)
    
    return tokens


# In[26]:

pos_lines = list(df[df.label == 1].headlines)

pos_tokens = process_text(pos_lines)
pos_freq = nltk.FreqDist(pos_tokens)

pos_freq.most_common(20)


# In[27]:


y_val = [x[1] for x in pos_freq.most_common()]
fig = plt.figure(figsize=(10,5))
plt.plot(y_val)
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Word Frequency Distribution (Positive)")
plt.show()


# In[28]:

y_final = []
for i, k, z, t in zip(y_val[0::4], y_val[1::4], y_val[2::4], y_val[3::4]):
    y_final.append(math.log(i + k + z + t))

x_val = [math.log(i + 1) for i in range(len(y_final))]

fig = plt.figure(figsize=(10,5))

plt.xlabel("Words (Log)")
plt.ylabel("Frequency (Log)")
plt.title("Word Frequency Distribution (Positive)")
plt.plot(x_val, y_final)
plt.show()


# In[29]:

neg_lines = list(df2[df2.label == -1].headlines)

neg_tokens = process_text(neg_lines)
neg_freq = nltk.FreqDist(neg_tokens)

neg_freq.most_common(20)


# In[30]:

y_val = [x[1] for x in neg_freq.most_common()]

fig = plt.figure(figsize=(10,5))
plt.plot(y_val)

plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Word Frequency Distribution (Negative)")
plt.show()


# In[31]:


y_final = []
for i, k, z in zip(y_val[0::3], y_val[1::3], y_val[2::3]):
    if i + k + z == 0:
        break
    y_final.append(math.log(i + k + z))

x_val = [math.log(i+1) for i in range(len(y_final))]

fig = plt.figure(figsize=(10,5))

plt.xlabel("Words (Log)")
plt.ylabel("Frequency (Log)")
plt.title("Word Frequency Distribution (Negative)")
plt.plot(x_val, y_final)
plt.show()


# In[43]:

## Extension 1.2: Network Node Map for Streaming Media from Reddit web scrape data 


# In[74]:

import plotly.offline as py
from plotly.graph_objs import *
from operator import itemgetter
import community
import networkx as nx
import colorlover as cl
import numpy as np
import pickle
from nltk.stem import WordNetLemmatizer, PorterStemmer
from string import punctuation
from collections import Counter
from operator import itemgetter
import community
from collections import OrderedDict
import re
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import matplotlib.pyplot as plt
from HTMLParser import HTMLParser
from bs4 import BeautifulSoup


# In[75]:

###Streaming Video
subreddit = reddit.subreddit('BestOfStreamingVideo')
top_subreddit = subreddit.top(limit=500)


# In[76]:

for submission in subreddit.top(limit=2):
    print(submission.title, submission.id)


# In[77]:

topics_dict = { "title":[],      
               "score":[], \
               "id":[], "url":[], 
                "comms_num": [], \
                "created": [], \
                "body":[]}


# In[78]:

for submission in top_subreddit:
    topics_dict["title"].append(submission.title)
    topics_dict["score"].append(submission.score)
    topics_dict["id"].append(submission.id)
    topics_dict["url"].append(submission.url)
    topics_dict["comms_num"].append(submission.num_comments)
    topics_dict["created"].append(submission.created)
    topics_dict["body"].append(submission.selftext)


# In[ ]:




# In[79]:

topics_data = pd.DataFrame(topics_dict)
topics_data.head()


# In[81]:

def get_date(created):
    return dt.datetime.fromtimestamp(created)
_timestamp = topics_data["created"].apply(get_date)
topics_data = topics_data.assign(timestamp = _timestamp)
#topics_data.to_csv('vidstream.csv') 
#topics_data.sentiment.polarity


# In[82]:

with open('vidstream.csv', 'r') as nodecsv:                    
    nodereader = csv.reader(nodecsv) 
    nodes = [n for n in nodereader][1:]                     

node_names = [n[0] for n in nodes]                                    

with open('vidstream.csv', 'r') as edgecsv:                         
    edgereader = csv.reader(edgecsv)                                   
    edges = [tuple(e) for e in edgereader][1:]     
print(len(node_names))


# In[83]:

print(len(node_names))
print(len(edges))
Gvid = nx.Graph()
Gvid.add_nodes_from(node_names)
Gvid.add_edges_from([(1,2),(1,3)])
density = nx.density(Gvid)
print("Network density:", density)


# In[84]:

Gvid.neighbors(1)


# In[85]:

#Gvid.add_edges_from(edges)
print(nx.info(Gvid))


# In[106]:

get_ipython().run_cell_magic('html', '', '<div id="d3-example"></div>\n<style>\n.node {stroke: #fff; stroke-width: 1.5px;}\n.link {stroke: #999; stroke-opacity: .6;}\n</style>')


# In[108]:

get_ipython().run_cell_magic('javascript', '', '// We load the d3.js library from the Web.\nrequire.config({paths:\n    {d3: "http://d3js.org/d3.v3.min"}});\nrequire(["d3"], function(d3) {\n  // The code in this block is executed when the\n  // d3.js library has been loaded.\n\n  // First, we specify the size of the canvas\n  // containing the visualization (size of the\n  // <div> element).\n  var width = 300, height = 300;\n\n  // We create a color scale.\n  var color = d3.scale.category10();\n\n  // We create a force-directed dynamic graph layout.\n  var force = d3.layout.force()\n    .charge(-120)\n    .linkDistance(30)\n    .size([width, height]);\n\n  // In the <div> element, we create a <svg> graphic\n  // that will contain our interactive visualization.\n  var svg = d3.select("#d3-example").select("svg")\n  if (svg.empty()) {\n    svg = d3.select("#d3-example").append("svg")\n          .attr("width", width)\n          .attr("height", height);\n  }\n\n  // We load the JSON file.\n  d3.json("graph.json", function(error, graph) {\n    // In this block, the file has been loaded\n    // and the \'graph\' object contains our graph.\n\n    // We load the nodes and links in the\n    // force-directed graph.\n    force.nodes(graph.nodes)\n      .links(graph.links)\n      .start();\n\n    // We create a <line> SVG element for each link\n    // in the graph.\n    var link = svg.selectAll(".link")\n      .data(graph.links)\n      .enter().append("line")\n      .attr("class", "link");\n\n    // We create a <circle> SVG element for each node\n    // in the graph, and we specify a few attributes.\n    var node = svg.selectAll(".node")\n      .data(graph.nodes)\n      .enter().append("circle")\n      .attr("class", "node")\n      .attr("r", 5)  // radius\n      .style("fill", function(d) {\n         // The node color depends on the club.\n         return color(d.club);\n      })\n      .call(force.drag);\n\n    // The name of each node is the node number.\n    node.append("title")\n        .text(function(d) { return d.name; });\n\n    // We bind the positions of the SVG elements\n    // to the positions of the dynamic force-directed\n    // graph, at each time step.\n    force.on("tick", function() {\n      link.attr("x1", function(d){return d.source.x})\n          .attr("y1", function(d){return d.source.y})\n          .attr("x2", function(d){return d.target.x})\n          .attr("y2", function(d){return d.target.y});\n\n      node.attr("cx", function(d){return d.x})\n          .attr("cy", function(d){return d.y});\n    });\n  });\n});')


# In[109]:

G=Gvid
G.graph
G.add_node(1, score='151')
G.add_nodes_from([3], score='28')
G.add_edge(1, 2, weight=4.7 )
G.add_edges_from([(3,4),(4,5)], color='red')
G.add_edges_from([(1,2,{'color':'blue'}), (2,3,{'weight':4})])
G[1][2]['weight'] = 4.7
G.edge[1][2]['weight'] = 4
nx.write_gml(G,"path.to.file")
mygraph=nx.read_gml("path.to.file")
fig, ax = plt.subplots(1, 1, figsize=(8, 6));
nx.draw_spectral(G, ax=ax)
nx.draw_networkx(G, ax=ax)
plt.show()
nx.write_gexf(G, 'Gvid.gexf')


# In[110]:

title_sig_dict = {}
score_dict = {}
id_dict = {}
url_dict = {}
comms_num_dict = {}
created_dict = {}
body_dict = {}
for node in nodes:
    title_sig_dict = {}
    score_dict[node[0]] = node[1]
    id_dict[node[0]] = node[2]
    url_dict[node[0]] = node[3]
    comms_num_dict[node[0]] = node[4]
    created_dict[node[0]] = node[5]
    body_dict[node[0]] = node[6]
G = nx.Graph(title="Status")
G.graph
G.add_node(1, score='151')
G.add_nodes_from([3], score='196')
G.add_edge(1, 2, weight=4.7 )
G.add_edges_from([(3,4),(4,5)], color='red')
G.add_edges_from([(1,2,{'color':'blue'}), (2,3,{'weight':8})])
G[1][2]['weight'] = 4.7
G.edge[1][2]['weight'] = 4
nx.write_gml(G,"path.to.file")
mygraph=nx.read_gml("path.to.file")
fig, ax = plt.subplots(1, 1, figsize=(8, 6));
nx.draw_spectral(G, ax=ax)
nx.draw_networkx(G, ax=ax)
plt.show()
nx.draw_circular(G)
plt.show()
nx.write_gexf(G, 'Gvid.gexf')


# In[ ]:



