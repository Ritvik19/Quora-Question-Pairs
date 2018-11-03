
# coding: utf-8

# In[1]:


import re, nltk, pickle
from scipy.sparse import hstack


# In[2]:


def clean_data(data, rsw, ps, wnl):
    try:
        remove_punc = lambda x : re.sub(r"[^A-Za-z,']", ' ', x)

        remove_extra_spaces = lambda x : re.sub(r"\s+", ' ', x)

        lower_case = lambda x : x.lower()

        stop_words = set(nltk.corpus.stopwords.words('english'))
        remove_stopwords = lambda x: ' '.join(word for word in x.split() if word not in stop_words)

        ps = nltk.stem.porter.PorterStemmer()
        ps_stem = lambda x: ' '.join(ps.stem(word) for word in x.split())

        wnl = nltk.stem.WordNetLemmatizer()
        wnl_lemmatize = lambda x: ' '.join(wnl.lemmatize(word) for word in x.split())
            
        data = remove_punc(data)
        data = remove_extra_spaces(data)
        data = lower_case(data)
            
        if rsw == True:
            data = remove_stopwords(data)
        if ps == True:
            data = ps_stem(data)
        if wnl == True:
            data = wnl_lemmatize(data)
         
        return data
    except NameError:
        print('Please Import pickle, re, nltk modules and from scipy.sparse import hstack')


# In[ ]:




