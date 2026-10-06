# Adult Income Practice Data Mining Project

Dataset: UCI Adult (census income). Each row is one person, described by
attributes such as age, job type, education and marital status. The last
column ("income") says whether that person earns >50K or <=50K per year.

Goals of this script

- Classification: teach a model to PREDICT the income class from the
  other attributes (decision tree, naive Bayes, kNN, SVM, neural network).
- Clustering: find groups of similar people WITHOUT using the income
  column (k-means).

Sections 0. Imports and settings

1. Load and clean the data
2. Helper function (per-class metrics)
3. Task 1: decision tree + naive Bayes (categorical attributes only)
4. Task 2 preprocessing: one-hot + mean-binarized numeric attributes
5. Task 2.1: k-means clustering (k = 3, 5, 10)
6. Task 2.2: kNN on the last 10 test records (k = 3, 5, 10)
7. Task 3: SVM
8. Task 4: neural network
9. Summary

Requires: pandas, numpy, scikit-learn (version 1.2 or newer)
pip install pandas numpy scikit-learn

Vocabulary used in the comments

- train data : rows the model learns from (answers are known).
- test data : rows held back to check how well the model works on
  people it has never seen.
- fit() : the "learning" step; the model studies the training data.
- predict() : the model guesses the class of new rows.
- feature : one input column the model looks at.
