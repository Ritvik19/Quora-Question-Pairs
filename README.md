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
  * abs_word_diff  
          
        absolute difference between number of words in question1 and question2
  * avg_text_len  
          
        average number of words present in the questions
  * rel_word_diff  
          
        abs_word_diff / avg_text_len
  * word_intersection  
          
        number of unique words common to question1 and question2
  * word_union: 
          
        number of unique words in question1 and question2 combined
  * jaccard_similarity
      
        |Q1 &cap; Q2| / |Q1 &cup; Q2|
___

**Performance Summary**

Features | Algorithm | Log Loss
:---:|:---:|:---:
Baseline | NA | 0.6585
Basic Features | Logistic Regression | 0.5807 (0.0013)
Basic Features | SGD Classifier | 0.6760 (0.1554)
