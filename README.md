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
          
        absolute difference between number of token in question1 and question2
  * avg_num_token  
          
        average number of tokens present in the questions
  * rel_token_diff  
          
        abs_token_diff / avg_num_token
  * token_intersection  
          
        number of unique tokens common to question1 and question2
  * token_union: 
          
        number of unique tokens in question1 and question2 combined
  * jaccard_similarity
      
        token_intersection / token_union
  * lcs_token
      
        length of longest common subsequence of tokens        
  * lcs_token_ratio
      
        lcs_token / avg_num_tokens
___

**Performance Summary**

Features | Algorithm | Log Loss
:---:|:---:|:---:
Baseline | NA | 0.6585
Basic Features | Logistic Regression | 0.5549 (0.0019)
Basic Features | SGD Classifier | 0.6108 (0.0292)
