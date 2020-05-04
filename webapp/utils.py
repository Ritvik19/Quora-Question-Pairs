import pickle
import numpy as np
import pandas as pd

import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup

from fuzzywuzzy import fuzz
from scipy.sparse import hstack
from scipy.spatial.distance import hamming, cosine

countVectorizer = pickle.load(open('BinaryCountVectorizer.pkl', 'rb'))
tfidfVectorizer = pickle.load(open('TfidfVectorizer.pkl', 'rb'))

tfidfModel = pickle.load(open('TFIDFVotingClassifier.pkl', 'rb'))
featuresModel = pickle.load(open('CustomFeaturesVotingClassifier.pkl', 'rb'))
classBalancedModel = pickle.load(open('ClassBalancedRF.pkl', 'rb'))

STOP_WORDS = stopwords.words("english")
ps = PorterStemmer()

def preprocess(x):
    x = str(x).lower()
    x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")\
                        .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
                        .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
                        .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
                        .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
                        .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                        .replace("€", " euro ").replace("'ll", " will")
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)
    
    x = re.sub(r'\W', ' ', str(x))
    
    x = re.sub(r"\s+", ' ', str(x))
    
    x = ' '.join(ps.stem(word) for word in x.split())
    
    soup = BeautifulSoup(str(x))
    x = soup.get_text()

    x = ' '.join([word for word in x.split() if word not in STOP_WORDS])
    
    return x

def lcs(X , Y): 
    m = len(X) 
    n = len(Y) 

    L = [[None]*(n+1) for i in range(m+1)] 

    for i in range(m+1): 
        for j in range(n+1): 
            if i == 0 or j == 0 : 
                L[i][j] = 0
            elif X[i-1] == Y[j-1]: 
                L[i][j] = L[i-1][j-1]+1
            else: 
                L[i][j] = max(L[i-1][j] , L[i][j-1]) 

    return L[m][n] 

def getweight(list_words):
    weight = 0
    for word in list_words:
        try:
            weight += tv.idf_[tv.vocabulary_[word]]
        except:
            pass
    return weight

def generateFeatures(text1, text2):

    abs_token_diff = abs(len(text1.split())-len(text2.split()))
    avg_num_token = (len(text1.split())+len(text2.split()))/2
    rel_token_diff = abs_token_diff/avg_num_token
    token_intersection = len(set(text1.split())&set(text2.split()))
    token_union = len(set(text1.split())|set(text2.split()))
    jaccard_similarity_token = token_intersection/token_union
    lcs_token = lcs(text1.split(), text2.split())
    lcs_token_ratio = lcs_token/avg_num_token
    
    text1_clean = preprocess(text1)
    text2_clean = preprocess(text2)
    
    abs_word_diff = abs(len(text1_clean.split())-len(text2_clean.split()))
    avg_num_word = (len(text1_clean.split())+len(text2_clean.split()))/2
    rel_word_diff = abs_word_diff/avg_num_word
    word_intersection = len(set(text1_clean.split())&set(text2_clean.split()))
    word_union = len(set(text1_clean.split())|set(text2_clean.split()))
    jaccard_similarity_word = word_intersection/word_union
    lcs_word = lcs(text1_clean.split(), text2_clean.split())
    lcs_word_ratio = lcs_word/avg_num_word
    
    fuzz_simple_ratio = fuzz.ratio(str(text1_clean).split(), str(text2_clean).split())
    fuzz_partial_ratio = fuzz.partial_ratio(str(text1_clean).split(), str(text2_clean).split())
    fuzz_token_sort_ratio = fuzz.token_sort_ratio(str(text1_clean).split(), str(text2_clean).split())
    fuzz_token_set_ratio = fuzz.token_set_ratio(str(text1_clean).split(), str(text2_clean).split())
    
    v1 = countVectorizer.transform([text1_clean])
    v2 = countVectorizer.transform([text2_clean])
    
    hamming_distance = hamming(v1.toarray()[0], v2.toarray()[0])
    
    v1 = tfidfVectorizer.transform([text1_clean])
    v2 = tfidfVectorizer.transform([text2_clean])
    
    cosine_distance = cosine(v1.toarray()[0], v2.toarray()[0])
    
    weighted_intersection = getweight(set(text1_clean.split())&set(text2_clean.split()))
    weighted_union = getweight(set(text1_clean.split())|set(text2_clean.split()))
    try:
        jaccard_similarity_weighted = weighted_intersection/weighted_union
    except:
        jaccard_similarity_weighted = -1
    
    features = {
        'abs_token_diff': abs_token_diff, 'avg_num_token': avg_num_token, 'rel_token_diff': rel_token_diff,
        'token_intersection': token_intersection, 'token_union': token_union, 
        'jaccard_similarity_token': jaccard_similarity_token, 'lcs_token': lcs_token, 'lcs_token_ratio': lcs_token_ratio,
        'abs_word_diff': abs_word_diff, 'avg_num_word': avg_num_word, 'rel_word_diff': rel_word_diff,
        'word_intersection': word_intersection, 'word_union': word_union, 
        'jaccard_similarity_word': jaccard_similarity_word, 'lcs_word': lcs_word, 'lcs_word_ratio': lcs_word_ratio,
        'fuzz_simple_ratio': fuzz_simple_ratio, 'fuzz_partial_ratio': fuzz_partial_ratio, 
        'fuzz_token_sort_ratio': fuzz_token_sort_ratio, 'fuzz_token_set_ratio': fuzz_token_set_ratio,
        'hamming_distance': hamming_distance, 'cosine_distance': cosine_distance,
        'weighted_intersection': weighted_intersection, 'weighted_union': weighted_union,
        'jaccard_similarity_weighted': jaccard_similarity_weighted
    }
    
    features = pd.DataFrame(features, index=[0])
    
    vects = hstack([v1, v2])
    
    return features, vects

def getPredictions(feats, vects):
    labels = {
        0: 'unique', 1: 'duplicate'
    }
    
    result = pd.DataFrame(np.concatenate([
        tfidfModel.estimators_[0].predict_proba(vects),
        tfidfModel.estimators_[1].predict_proba(vects),
        tfidfModel.estimators_[2].predict_proba(vects),
        tfidfModel.estimators_[3].predict_proba(vects),
        featuresModel.estimators_[0].predict_proba(feats),
        featuresModel.estimators_[1].predict_proba(feats),
        featuresModel.estimators_[2].predict_proba(feats),
        classBalancedModel.predict_proba(feats),
    ], axis=0), index=['LR vects 1', 'LR vects 10', 'LR vects 100', 'SGD vects', 'LR', 'XGB', 'GB', 'RF'], 
                                columns=['prob_unique', 'prob_duplicate'])
    
    
    
    result['prediction'] = result['prob_duplicate'].apply(lambda x : labels[round(x)])
    result['prob_duplicate'] = result['prob_duplicate'].apply(lambda x : round(x*100, 2))
    result['prob_unique'] = result['prob_unique'].apply(lambda x : round(x*100, 2))
    
    
    return result.to_dict()