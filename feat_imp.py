#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 13:26:28 2022

@author: vedjain
"""
def write_pickle(obj_in, path_in, file_n):
    import pickle
    pickle.dump(obj_in, open(path_in + file_n + ".pk", "wb"))
    
def read_pickle(path_o, file_n):
    import pickle
    my_pd_t = pickle.load(open(path_o + file_n + ".pk", "rb"))
    return my_pd_t

def rem_url(var_in):
    import re
    return re.sub('http://\S+|https://\S+', '', var_in)

def clean_txt(var_in):
    import re
    import contractions
    #pip install contractions
    #expands contractions
    var_in = rem_url(str(var_in))
    var_in = contractions.fix(var_in)
    tmp = re.sub("[^A-Za-z!?]+", " ", var_in).lower()
    return tmp

def rem_sw(var_in):
    import nltk
    from nltk.corpus import stopwords
    sw = list(set(stopwords.words('english')))
    sw.append('like')
    sw.append('comment')
    sw.append('subscribe')
    sw.append('video')
    tmp = [word for word in var_in if word not in sw]
    return tmp

def rem_spec_words(var_in):
    sw = list(('youtube','video','official','follow','instagram','facebook','twitter','channel','tiktok','video','get', 'like', 'comment', 'subscribe', '?', '!', 'would'))
    tmp = [word for word in var_in if word not in sw]
    return (tmp)


from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

# import statsmodels.formula.api as smf

out_path = "/Users/vedjain/python_npl/youtube_project/"

df = read_pickle(out_path, "df")


# df['tags'] = df.tags.str.join(' ') #untokenize the tags
# df['tags'] = df.tags.apply(clean_txt)
# df.dropna(subset="tags")
df = df.reset_index()
df['tokenized_title_sw'] = df.tokenized_title_sw.apply(rem_spec_words)
df['tokenized_description_sw'] = df.tokenized_description_sw.apply(rem_spec_words)

features_title = pd.DataFrame()
features_description = pd.DataFrame()

features_title['tokens'] = ""

features_description['tokens'] = ""

words = df['tokenized_title_sw'][0]
views = list()
like = list()
com = list()
for i in words:
    views.append(df['views_bucket'][0])
    like.append(df['likes_bucket'][0])
    com.append(df['comments_bucket'][0])

for k, i in enumerate(df['tokenized_title_sw'][1:]):
    words += i
    length = len(i)
    for x in range(length):
        views.append(df['views_bucket'][k])
        like.append(df['likes_bucket'][k])
        com.append(df['comments_bucket'][k])
    # words.append(df['views_bucket'][k])
    
features_title['tokens'] = words
features_title['views_bucket'] = views
features_title['likes_bucket'] = like
features_title['comments_bucket'] = com

words = df['tokenized_description_sw'][0]
views = list()
like = list()
com = list()

for i in words:
    views.append(df['views_bucket'][0])
    like.append(df['likes_bucket'][0])
    com.append(df['comments_bucket'][0])


for k, i in enumerate(df['tokenized_description_sw'][1:]):
    words += i
    length = len(i)
    for x in range(length):
        views.append(df['views_bucket'][k])
        like.append(df['likes_bucket'][k])
        com.append(df['comments_bucket'][k])

features_description['tokens'] = words
features_description['views_bucket'] = views
features_description['likes_bucket'] = like
features_description['comments_bucket'] = com


features_description['views_bucket'] = features_description['views_bucket'].str[-1:]
features_title['views_bucket'] = features_title['views_bucket'].str[-1:]

features_description['likes_bucket'] = features_description['likes_bucket'].str[-1:]
features_title['likes_bucket'] = features_title['likes_bucket'].str[-1:]

features_description['comments_bucket'] = features_description['comments_bucket'].str[-1:]
features_title['comments_bucket'] = features_title['comments_bucket'].str[-1:]

features_title.drop_duplicates(subset='tokens', inplace=True)
features_description.drop_duplicates(subset='tokens', inplace=True)



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

print('Libraries Imported')

"""
TITLE
"""

features_title['num_rep'] = features_title['tokens']
factor = pd.factorize(features_title['num_rep'])
features_title.num_rep = factor[0]
definitions = factor[1]
print(features_title.num_rep.head())
print(definitions)

X = features_title['num_rep']
y = features_title['views_bucket']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 21)

scaler = StandardScaler()
X_train = np.array(X_train).reshape(-1,1)
X_test = np.array(X_test).reshape(-1,1)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
compare_title_views = (pd.crosstab(y_test, y_pred, rownames=['Actual Bucket'], colnames=['Predicted Bucket']))

comparing_title = pd.DataFrame()
comparing_title['actual_views'] = y_test
comparing_title['prediction_views'] = y_pred

title_views_acc = (classification_report(y_test, y_pred))

X = features_title['num_rep']
y = features_title['comments_bucket']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 21)

scaler = StandardScaler()
X_train = np.array(X_train).reshape(-1,1)
X_test = np.array(X_test).reshape(-1,1)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
compare_title_comments = (pd.crosstab(y_test, y_pred, rownames=['Actual Bucket'], colnames=['Predicted Bucket']))

comparing_title['actual_comments'] = y_test
comparing_title['prediction_comments'] = y_pred

title_comments_acc = (classification_report(y_test, y_pred))

X = features_title['num_rep']
y = features_title['likes_bucket']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 21)

scaler = StandardScaler()
X_train = np.array(X_train).reshape(-1,1)
X_test = np.array(X_test).reshape(-1,1)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
compare_title_likes = (pd.crosstab(y_test, y_pred, rownames=['Actual Bucket'], colnames=['Predicted Bucket']))

comparing_title['actual_likes'] = y_test
comparing_title['prediction_likes'] = y_pred

comparing_title.index = features_title['tokens'][comparing_title.index]

title_likes_acc = (classification_report(y_test, y_pred))


"""
DESCRIPTION
"""
features_description['num_rep'] = features_description['tokens']
factor = pd.factorize(features_description['num_rep'])
features_description.num_rep = factor[0]
definitions = factor[1]
print(features_description.num_rep.head())
print(definitions)

X = features_description['num_rep']
y = features_description['views_bucket']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 21)

scaler = StandardScaler()
X_train = np.array(X_train).reshape(-1,1)
X_test = np.array(X_test).reshape(-1,1)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
compare_description_views = (pd.crosstab(y_test, y_pred, rownames=['Actual Bucket'], colnames=['Predicted Bucket']))

comparing_description = pd.DataFrame()
comparing_description['actual_views'] = y_test
comparing_description['prediction_views'] = y_pred

description_views_acc = (classification_report(y_test, y_pred))

X = features_description['num_rep']
y = features_description['comments_bucket']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 21)

scaler = StandardScaler()
X_train = np.array(X_train).reshape(-1,1)
X_test = np.array(X_test).reshape(-1,1)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
compare_description_comments = (pd.crosstab(y_test, y_pred, rownames=['Actual Bucket'], colnames=['Predicted Bucket']))

comparing_description['actual_comments'] = y_test
comparing_description['prediction_comments'] = y_pred

description_comments_acc = (classification_report(y_test, y_pred))

X = features_description['num_rep']
y = features_description['likes_bucket']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 21)

scaler = StandardScaler()
X_train = np.array(X_train).reshape(-1,1)
X_test = np.array(X_test).reshape(-1,1)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
compare_description_likes = (pd.crosstab(y_test, y_pred, rownames=['Actual Bucket'], colnames=['Predicted Bucket']))

comparing_description['actual_likes'] = y_test
comparing_description['prediction_likes'] = y_pred

comparing_description.index = features_description['tokens'][comparing_description.index]

description_likes_acc = (classification_report(y_test, y_pred))
