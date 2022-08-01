#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 10:56:22 2022

@author: vedjain
"""
def rem_sw(var_in):
    import nltk
    from nltk.corpus import stopwords
    sw = list(set(stopwords.words('english')))
    sw.append("like")
    sw.append("comment")
    sw.append("subscribe")
    tmp = [word for word in var_in.split() if word not in sw]
    return ' '.join(tmp)

def stem_fun(var_in):
    from nltk.stem.porter import PorterStemmer
    ps = PorterStemmer()
    ps_res = [ps.stem(word) for word in var_in.split()]
    # ps_res = list()
    # for word_t in var_in.split():
    #     ps_res.append(ps.stem(word_t))
    return ' '.join(ps_res)

def clean_txt(var_in):
    import re
    import contractions
    #pip install contractions
    #expands contractions
    var_in = rem_url(str(var_in))
    var_in = contractions.fix(var_in)
    tmp = re.sub("[^A-Za-z!?]+", " ", var_in).lower()
    return tmp

def rem_url(var_in):
    import re
    return re.sub('http://\S+|https://\S+', '', var_in)

def get_time(var_in):
    return var_in[-10:]

def time_diff (d1, d2):
    from dateutil import parser
    date1 = parser.parse(d1)
    date2 = parser.parse(d2)
    
    return date2 - date1

def stem_fun(var_in):
    from nltk.stem.porter import PorterStemmer
    ps = PorterStemmer()
    ps_res = [ps.stem(word) for word in var_in.split()]
    # ps_res = list()
    # for word_t in var_in.split():
    #     ps_res.append(ps.stem(word_t))
    return ' '.join(ps_res)

#-----------------------------------------------------

import pandas as pd
import nltk
data_path = "/Users/vedjain/python_npl/youtube_project/data/"


#read in data
df = pd.read_csv('US_youtube_trending_data.csv')

#drop all videos besides ones posted in this year
df.drop(df[df['publishedAt'] < '2022-01-01'].index, inplace = True)

#sort by descending order of when it was trending
df.sort_values(by=["trending_date"], ascending = False, inplace=True)

#reset the indices
df = df.reset_index(drop=True) #update indicies after sort


df["title_clean"] = df.title.apply(clean_txt)
df["description_clean"] = df.description.apply(clean_txt)

df["title_sw"] = df.title_clean.apply(rem_sw)
df["description_sw"] = df.description_clean.apply(rem_sw)

df['title_stem'] = df.title_sw.apply(stem_fun)
df['description_stem'] = df.description_sw.apply(stem_fun)

#tokenize the tags
df['tags'] = df.tags.str.split(pat = '|')

df['publish_time'] = df.publishedAt.apply(get_time)
df['trending_time'] = df.trending_date.apply(get_time)

df['publishedAt'] = df['publishedAt'].str[0:10]
df['trending_date'] = df['trending_date'].str[0:10]

df['time_diff'] = df.apply(lambda x: time_diff(x['publishedAt'], x['trending_date']), axis=1)


#tokenizing the sw for title and description

#import nltk
#nltk.download('punkt')
df['tokenized_title_sw'] = df['title_sw'].apply(nltk.word_tokenize)
df['tokenized_description_sw'] = df['description_sw'].apply(nltk.word_tokenize)
