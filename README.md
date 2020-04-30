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
    * hamming distance
    * cosine distance

* TFIDF Features
___

**Performance Summary**

Features | Algorithm | Log Loss | TP% | TN% 
:---:|:---:|:---:
Baseline | NA | 0.6585 | NA | NA
Basic Features | Logistic Regression | 0.5829 (0.0012) | 0.60 | 0.78
Basic Features Cleaned | Logistic Regression | 0.5580 (0.0020) | 0.62 | 0.79
Fuzzy Features | Logistic Regression | 0.5529 (0.0022) | 0.62 | 0.79
Advanced Features | Logistic Regression | 0.5509 (0.0017) | 0.64 | 0.79
TFIDF Features | Logistic Regression | 0.5423 (0.0025) | 0.74 | 0.71
TFIDF Features | Multinomial NB | 0.5621 (0.0032) | 0.88 | 0.49
