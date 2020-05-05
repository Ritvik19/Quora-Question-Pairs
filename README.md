# Quora-Question-Pairs

Quora is a place to gain and share knowledge—about anything. It’s a platform to ask questions and connect with people who contribute unique insights and quality answers. This empowers people to learn from each other and to better understand the world.

Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.

Source: [Kaggle](https://www.kaggle.com/c/quora-question-pairs)

**Problem Statement**: Classify whether question pairs are duplicates or not

**Project  Objective**: Build model that can indentify similar texts
___

**Features Summary**

* Baseline Model: 

      predicting every point with the mean value
* Basic Features:
  * abs_token_diff  
          
        absolute difference between number of tokens in question1 and question2
  * avg_num_token  
          
        average number of tokens present in the questions
  * rel_token_diff  
          
        abs_token_diff / avg_num_token
  * token_intersection  
          
        number of unique tokens common to question1 and question2
  * token_union: 
          
        number of unique tokens in question1 and question2 combined
  * jaccard_similarity_token
      
        token_intersection / token_union
  * lcs_token
      
        length of longest common subsequence of tokens        
  * lcs_token_ratio
      
        lcs_token / avg_num_token

* Text Cleaning
  * lower case
  * expanding contractions
  * remove unnecessary characters
  * stemming
  * removing markup
  * removing stopwords

* Basic Features Cleaned:
  * abs_word_diff  
          
        absolute difference between number of words in question1 and question2
  * avg_num_word 
          
        average number of words present in the questions
  * rel_word_diff  
          
        abs_word_diff / avg_num_word
  * word_intersection  
          
        number of unique words common to question1 and question2
  * word_union:
          
        number of unique words in question1 and question2 combined
  * jaccard_similarity_word
      
        word_intersection / word_union
  * lcs_word
      
        length of longest common subsequence of words        
  * lcs_word_ratio
      
        lcs_word / avg_num_word        

* Fuzzy Features
    * fuzz_simple_ratio
    * fuzz_partial_ratio
    * fuzz_token_sort_ratio
    * fuzz_token_set_ratio

* Advanced Features
  * hamming_distance
      
        hamming distance between binary count vectors of cleaned texts   
  * cosine_distance
      
        cosine distance between tfidf vectors of cleaned texts  
  * weighted_intersection
      
        weight of unique words common to question1 and question2 
  * weighted_union
      
        weight of unique words in question1 and question2 combined 
  * jaccard_similarity_weighted
      
        weighted_intersection / weighted_union


* TFIDF Features
    
    * tfidf vectors
___

**Performance Summary**

Features | Algorithm | Log Loss | FP% | FN% 
:---|:---|---:|---:|---:
Baseline | NA | 0.6585 | NA | NA
Basic Features | Logistic Regression | 0.5829 (0.0012) | 0.40 | 0.22
Basic Features Cleaned | Logistic Regression | 0.5580 (0.0020) | 0.38 | 0.21
Fuzzy Features | Logistic Regression | 0.5529 (0.0022) | 0.38 | 0.21
Advanced Features | Logistic Regression | 0.5509 (0.0017) | 0.36 | 0.21
Advanced Features | XGBoost Classifier | 0.4569 (0.0012) | 0.23 | 0.27
Advanced Features | Random Forest Classifier | 0.8140 (0.0139) | 0.17 | 0.41
Advanced Features | Gradient Boosting Classifier | 0.4688 (0.0011) | 0.24 | 0.28
Advanced Features | Bagging Classifier | 0.5509 (0.0017) | 0.37 | 0.21
Advanced Features | SGD Classifier | 0.5928 (0.0304) | 0.33 | 0.33
Advanced Features | Voting Classifier (LR, XGB, GB) | 0.4782 (0.0010) | 0.28 | 0.23
TFIDF Features | Logistic Regression C1| 0.5322 (0.0015) | 0.26 | 0.28
TFIDF Features | Logistic Regression C10| 0.5301 (0.0015) | 0.28 | 0.25
TFIDF Features | Logistic Regression C100| 0.5474 (0.0026) | 0.31 | 0.23
TFIDF Features | Multinomial NB | 0.5450 (0.0019) | 0.14 | 0.46
TFIDF Features | SGD Classifier | 0.5543 (0.0042) | 0.29 | 0.27
TFIDF Features | Voting Classifier (LR1, LR10, LR100, SGD) | 0.5278 (0.0017) | 0.28 | 0.25
Over Sampling | Logistic Regression | 0.5329 (0.0021) | 0.36 | 0.20
Over Sampling | XGBoost Classifier | 0.4542 (0.0020) | 0.34 | 0.10
Over Sampling | Gradient Boosting Classifier | 0.4495 (0.0026) | 0.33 | 0.11
Over Sampling | Bagging Classifier | 0.5483 (0.0019) | 0.37 | 0.21
Over Sampling | Random Forest Classifier | 0.5851 (0.0102) | 0.20 | 0.092
Over Sampling | Random Forest Classifier 25 | 0.4066 (0.0083) | 0.23 | 0.058
Over Sampling | Random Forest Classifier 50 | 0.3713 (0.0073) | 0.23 | 0.058
Over Sampling | Random Forest Classifier 100 | 0.3504 (0.0059) | 0.23 | 0.054
Over Sampling | Random Forest Classifier 150 | 0.3416 (0.0051) | 0.23 | 0.054
Over Sampling | Random Forest Classifier 250 | 0.3336 (0.0040) | 0.23 | 0.053
Over Sampling | Random Forest Classifier 500 | 0.3260 (0.0037) | 0.23 | 0.053
