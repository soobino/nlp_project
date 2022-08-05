#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 22:17:41 2022

@author: soobino
"""

def read_pickle(path_o, file_n):
    import pickle
    my_pd_t = pickle.load(open(path_o + file_n + ".pk", "rb"))
    return my_pd_t

def wrd_freq_fun_redux(df_in, name_in):
    import collections
    word_dictionary = dict()
    for word in df_in.category.unique():
        tmp = df_in[df_in.category == word]
        tmp_txt = tmp[name_in].str.cat(sep=" ")
        tmp_wrd_frq = collections.Counter(tmp_txt.split())
        word_dictionary[word] = tmp_wrd_frq
    return word_dictionary

def lda_fun(df_in, n_topics_in, num_words_in, path_o):
    import gensim
    import gensim.corpora as corpora
    #from gensim.models.coherencemodel import CoherenceModel
    
    data_tmp = df_in.str.split()
    id2word = corpora.Dictionary(data_tmp)
    
    corpus = [id2word.doc2bow(text) for text in data_tmp]

    ldamodel = gensim.models.ldamodel.LdaModel(
        corpus, num_topics=n_topics_in, id2word=id2word, passes=15)
    #ldamodel.save(path_o + 'model5.gensim')
    topics = ldamodel.print_topics(num_words=num_words_in)
    for topic in topics:
        print(topic)
    return topics
###
import pandas as pd 

out_path = "/Users/soobino/Documents/School/5067 - NLP/project/"

df= read_pickle(out_path, 'df')

# word frequency 
df_freq_title = wrd_freq_fun_redux(df, "title_sw")

df_freq_desc = wrd_freq_fun_redux(df, "description_sw")

# lda
lda_title = lda_fun(df.title_sw, 15, 3, out_path)

lda_desc = lda_fun(df.title_sw, 15, 3, out_path)

cat_lda_title = pd.DataFrame()

for word in df.category.unique():
    tmp = df[df.category == word]
    lda = lda_fun(tmp.title_sw, 3, 3, out_path)
    cat_lda_title = cat_lda_title.append({"Category": word, "lda": lda}, ignore_index=True)
    
cat_lda_desc = pd.DataFrame()

for word in df.category.unique():
    tmp = df[df.category == word]
    lda = lda_fun(tmp.description_sw, 3, 3, out_path)
    cat_lda_desc = cat_lda_title.append({"Category": word, "lda": lda}, ignore_index=True)
     
import numpy as np
avg_perf = df.groupby('category')['likes','view_count','comment_count'].mean()

like_summary = df.groupby('category')['likes'].agg([np.mean, np.std, np.var])
view_summary = df.groupby('category')['view_count'].agg([np.mean, np.std, np.var])
comment_summary = df.groupby('category')['comment_count'].agg([np.mean, np.std, np.var])

    
    
