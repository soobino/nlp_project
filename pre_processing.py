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

def write_pickle(obj_in, path_in, file_n):
    import pickle
    pickle.dump(obj_in, open(path_in + file_n + ".pk", "wb"))
    
def read_pickle(path_o, file_n):
    import pickle
    my_pd_t = pickle.load(open(path_o + file_n + ".pk", "rb"))
    return my_pd_t
#-----------------------------------------------------

import pandas as pd
import nltk
import numpy as np

data_path = "/Users/vedjain/python_npl/youtube_project/data/"
out_path = "/Users/vedjain/python_npl/youtube_project/"

#read in data
df = pd.read_csv('US_youtube_trending_data.csv')

#drop all videos besides ones posted in this year
df.drop(df[df['publishedAt'] < '2022-01-01'].index, inplace = True)
#sort by descending order of when it was trending

cols = ['video_id','title']
df['num_days_trending'] = df.groupby(cols)['video_id'].transform('size')

df.drop_duplicates(subset = ['video_id'], inplace = True)

df.sort_values(by=["trending_date"], ascending = False, inplace=True)

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

#define array of data
data = np.array(df['view_count'].tolist())

view1, view2, view3, view4 = np.percentile(data, [20, 40, 60, 80])

data = np.array(df['likes'].tolist())

likes1, likes2, likes3, likes4 = np.percentile(data, [20, 40, 60, 80])

data = np.array(df['comment_count'].tolist())

com1, com2, com3, com4 = np.percentile(data, [20, 40, 60, 80])

conditions = [
    (df['view_count'] <= view1),
    (df['view_count'] > view1) & (df['view_count'] <= view2),
    (df['view_count'] > view2) & (df['view_count'] <= view3),
    (df['view_count'] > view3) & (df['view_count'] <= view4),
    (df['view_count'] > view4)
    ]

# create a list of the values we want to assign for each condition
values = ['bucket_1', 'bucket_2', 'bucket_3', 'bucket_4', 'bucket_5']

# create a new column and use np.select to assign values to it using our lists as arguments
df['views_bucket'] = np.select(conditions, values)

conditions = [
    (df['likes'] <= likes1),
    (df['likes'] > likes1) & (df['likes'] <= likes2),
    (df['likes'] > likes2) & (df['likes'] <= likes3),
    (df['likes'] > likes3) & (df['likes'] <= likes4),
    (df['likes'] > likes4)
    ]

# create a list of the values we want to assign for each condition
values = ['bucket_1', 'bucket_2', 'bucket_3', 'bucket_4', 'bucket_5']

# create a new column and use np.select to assign values to it using our lists as arguments
df['likes_bucket'] = np.select(conditions, values)

conditions = [
    (df['comment_count'] <= com1),
    (df['comment_count'] > com1) & (df['comment_count'] <= com2),
    (df['comment_count'] > com2) & (df['comment_count'] <= com3),
    (df['comment_count'] > com3) & (df['comment_count'] <= com4),
    (df['comment_count'] > com4)
    ]

# create a list of the values we want to assign for each condition
values = ['bucket_1', 'bucket_2', 'bucket_3', 'bucket_4', 'bucket_5']

# create a new column and use np.select to assign values to it using our lists as arguments
df['comments_bucket'] = np.select(conditions, values)


write_pickle(df, out_path, "df")
