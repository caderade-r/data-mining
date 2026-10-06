"""
Adult Income Practice Data Mining Project
================================
Dataset: UCI Adult (census income). Each row is one person, described by
attributes such as age, job type, education and marital status. The last
column ("income") says whether that person earns >50K or <=50K per year.

Goals of this script
  * Classification: teach a model to PREDICT the income class from the
    other attributes (decision tree, naive Bayes, kNN, SVM, neural network).
  * Clustering: find groups of similar people WITHOUT using the income
    column (k-means).

Sections
  0. Imports and settings
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
  * train data : rows the model learns from (answers are known).
  * test data  : rows held back to check how well the model works on
                 people it has never seen.
  * fit()      : the "learning" step; the model studies the training data.
  * predict()  : the model guesses the class of new rows.
  * feature    : one input column the model looks at.
"""

# ---------------------------------------------------------------------
# Imports and settings
# ---------------------------------------------------------------------
import os                      # create folders / build file paths
import time                    # time how long slow models take to train
import warnings                # hide harmless warning messages

import numpy as np             # numerical arrays and math helpers
import pandas as pd            # tables ("DataFrames") for loading/cleaning data

# One-hot encoder: turns text categories into 0/1 columns.
from sklearn.preprocessing import OneHotEncoder
# Decision tree: learns a flowchart of yes/no questions.
from sklearn.tree import DecisionTreeClassifier
# Naive Bayes for binary (0/1) features: classifies using probabilities.
from sklearn.naive_bayes import BernoulliNB
# K-means: groups similar rows into k clusters (no labels needed).
from sklearn.cluster import KMeans
# kNN: classifies a row by a vote among its k most similar training rows.
from sklearn.neighbors import KNeighborsClassifier
# SVM: finds the best boundary separating the two classes.
from sklearn.svm import SVC
# Neural network (multi-layer perceptron): layers of weighted calculations.
from sklearn.neural_network import MLPClassifier
# Metrics: confusion matrix counts right/wrong guesses per class;
# accuracy_score is the fraction of correct guesses.
from sklearn.metrics import confusion_matrix, accuracy_score

# Hide non-critical warnings (e.g. neural network "did not converge") so the
# console output stays clean for screenshots.
warnings.filterwarnings("ignore", category=UserWarning)

# A fixed "random seed" makes results repeatable: several algorithms start
# from random choices, and the same seed gives the same results every run.
RANDOM_STATE = 0

# Folder where the k-means centroid tables will be saved as CSV files.
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)   # create it if it doesn't exist yet

# Make printed tables wider so columns are not cut off in screenshots.
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

# Where the data lives. The script downloads it directly from UCI.
BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
# If you downloaded the files, point these at your local copies instead, e.g.
# TRAIN_PATH = "data/adult.data";  TEST_PATH = "data/adult.test"
TRAIN_PATH = BASE_URL + "adult.data"
TEST_PATH = BASE_URL + "adult.test"

# The raw files have no header row, so we supply the column names
# (taken from the adult.names description file).
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income",
]
# Continuous (numeric) attributes: numbers that can take many values.
CONTINUOUS = ["age", "fnlwgt", "education-num", "capital-gain",
              "capital-loss", "hours-per-week"]
# Categorical attributes: every column that is neither numeric nor the label.
CATEGORICAL = [c for c in COLUMNS if c not in CONTINUOUS + ["income"]]
# The two possible values of the class we are predicting.
CLASS_LABELS = ["<=50K", ">50K"]

# ---------------------------------------------------------------------
# 1. Load and clean the data
# ---------------------------------------------------------------------
print("=" * 60)
print("1. LOADING AND CLEANING DATA")
print("=" * 60)

# pd.read_csv reads a comma-separated text file into a table.
#   names=COLUMNS          -> use our column names (file has no header)
#   skipinitialspace=True  -> ignore the space after each comma
#   na_values="?"          -> treat '?' as "missing value" (NaN)
train = pd.read_csv(TRAIN_PATH, names=COLUMNS, skipinitialspace=True,
                    na_values="?")
# adult.test has a junk first line, so skiprows=1 skips it.
test = pd.read_csv(TEST_PATH, names=COLUMNS, skipinitialspace=True,
                   na_values="?", skiprows=1)

# .shape gives (number of rows, number of columns).
print(f"Before cleaning: train={train.shape}, test={test.shape}")

# Test labels end with a '.', e.g. '>50K.'. Remove the dot so the labels
# match the training labels; otherwise every prediction would look wrong.
test["income"] = test["income"].str.rstrip(".")

# dropna() removes every row containing at least one missing ('?') value,
# as the assignment requires. reset_index(drop=True) renumbers the rows
# 0, 1, 2, ... after the removal.
train = train.dropna().reset_index(drop=True)
test = test.dropna().reset_index(drop=True)

print(f"After removing '?' records: train={train.shape}, test={test.shape}")
# value_counts() counts how many rows belong to each class.
print("Train class counts:\n", train["income"].value_counts())

# The "answers" (labels) we want the models to predict.
y_train = train["income"]
y_test = test["income"]


# ---------------------------------------------------------------------
# 2. Helper: per-class metrics
# ---------------------------------------------------------------------
def per_class_report(y_true, y_pred, labels=CLASS_LABELS):
    """Build a table of TP rate, FP rate, precision, recall and F1 per class.

    How it works:
      1. Compare the true labels with the model's predictions using a
         confusion matrix (a grid counting right and wrong guesses).
      2. For each class in turn, treat that class as the "positive" class
         and count:
           TP (true positive)  : predicted this class, and it was right
           FN (false negative) : it was this class, but the model missed it
           FP (false positive) : predicted this class, but it was wrong
           TN (true negative)  : correctly predicted "not this class"
      3. Compute the metrics:
           TP rate (= recall) : TP / (TP + FN)  how many real cases were caught
           FP rate            : FP / (FP + TN)  how many non-cases were wrongly flagged
           precision          : TP / (TP + FP)  when the model says this class,
                                                how often is it right
           F1                 : balance of precision and recall

    Parameters
      y_true : the real labels
      y_pred : the model's predicted labels
      labels : class names, in the order used for the confusion matrix

    Returns a pandas table with one row per class.
    """
    # Rows of cm = true class, columns = predicted class.
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    for i, label in enumerate(labels):
        tp = cm[i, i]                        # correct guesses for this class
        fn = cm[i, :].sum() - tp             # this class, but missed
        fp = cm[:, i].sum() - tp             # other class, wrongly labeled this
        tn = cm.sum() - tp - fn - fp         # everything else
        # The "if" parts avoid dividing by zero.
        tp_rate = tp / (tp + fn) if (tp + fn) else 0.0   # same as recall
        fp_rate = fp / (fp + tn) if (fp + tn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp_rate
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        rows.append([label, tp_rate, fp_rate, precision, recall, f1])
    return pd.DataFrame(
        rows,
        columns=["class", "TP rate", "FP rate", "precision", "recall", "F1"],
    ).round(4)


# ---------------------------------------------------------------------
# 3. Task 1: categorical attributes only
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. TASK 1: DECISION TREE AND NAIVE BAYES (categorical only)")
print("=" * 60)

# One-hot encoding turns each category into its own yes/no (1/0) column.
# Example: occupation = "Sales" becomes occupation_Sales=1 and every other
# occupation_* column = 0. Models need numbers, and this avoids implying that
# one category is "bigger" than another.
#   handle_unknown="ignore" -> if the test data contains a category never
#                              seen in training, give it all zeros instead
#                              of crashing.
#   sparse_output=False     -> return an ordinary table of numbers.
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

# fit_transform on TRAIN: the encoder learns which categories exist, then
# converts the training data. transform on TEST: reuse the same categories,
# so both tables have identical columns. We never fit on test data, because
# that would let the test set influence the setup ("data leakage").
X_train_cat = encoder.fit_transform(train[CATEGORICAL])
X_test_cat = encoder.transform(test[CATEGORICAL])
print(f"One-hot feature matrix: train={X_train_cat.shape}, "
      f"test={X_test_cat.shape}")

# --- 3.1 Decision tree ---
# A decision tree learns a flowchart of yes/no questions about the features
# (like "20 Questions") and gives a guess at the end of each path.
tree = DecisionTreeClassifier(
    criterion="entropy",       # pick each question by information gain: the
                               # best question splits people into purer groups
    max_depth=10,              # allow at most 10 questions in a row; without a
                               # limit the tree memorizes the training data
                               # ("overfitting") and does worse on new people
    random_state=RANDOM_STATE, # repeatable results
)
tree.fit(X_train_cat, y_train)         # learn the tree from the training data
tree_pred = tree.predict(X_test_cat)   # guess the class of every test person

print("\n--- Decision Tree (criterion=entropy, max_depth=10) ---")
print(per_class_report(y_test, tree_pred).to_string(index=False))
# accuracy_score = fraction of test people classified correctly overall.
tree_acc_task1 = accuracy_score(y_test, tree_pred)
print(f"Overall accuracy: {tree_acc_task1:.4f}")

# --- 3.2 Naive Bayes ---
# Naive Bayes uses probability: for a person with certain features, how
# likely is >50K compared with <=50K? It treats every feature as
# independent of the others (the "naive" assumption), which is not exactly
# true but works surprisingly well and is very fast.
# BernoulliNB is the version designed for binary (0/1) features like ours.
nb = BernoulliNB()
nb.fit(X_train_cat, y_train)           # learn the probabilities from train data
nb_pred = nb.predict(X_test_cat)       # pick the more likely class per person

print("\n--- Naive Bayes (BernoulliNB) ---")
print(per_class_report(y_test, nb_pred).to_string(index=False))
nb_acc_task1 = accuracy_score(y_test, nb_pred)
print(f"Overall accuracy: {nb_acc_task1:.4f}")

# ---------------------------------------------------------------------
# 4. Task 2 preprocessing: one-hot + mean-binarized numeric attributes
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. TASK 2 PREPROCESSING")
print("=" * 60)

# For Task 2 we keep the numeric attributes too, but convert each to 0/1:
# 1 if the person's value is ABOVE the average, 0 otherwise.
# The average is computed on TRAIN only (again to avoid data leakage) and
# the same averages are applied to the test data.
train_means = train[CONTINUOUS].mean()
print("Train means used for binarizing:\n", train_means.round(2))


def build_features(df):
    """Convert a cleaned table of people into an all-binary (0/1) feature table.

    Steps:
      1. Each numeric column becomes 1 if the value is greater than that
         column's TRAIN mean, else 0 (e.g. age_gt_mean).
      2. Each categorical column is one-hot encoded with the encoder that
         was already fitted on the training data (so the columns match).
      3. The two parts are placed side by side.

    Parameters
      df : a cleaned train or test table (with the original columns)

    Returns a table of 0/1 values, one column per feature. Because every
    column is 0/1, distance calculations (k-means, kNN) treat all features
    on the same scale.
    """
    # (value > mean) gives True/False; astype(int) turns that into 1/0.
    numeric_bin = (df[CONTINUOUS] > train_means).astype(int)
    numeric_bin.columns = [c + "_gt_mean" for c in CONTINUOUS]
    # encoder.transform produces the one-hot columns; get_feature_names_out
    # gives them readable names such as "occupation_Sales".
    cat_bin = pd.DataFrame(
        encoder.transform(df[CATEGORICAL]),
        columns=encoder.get_feature_names_out(CATEGORICAL),
        index=df.index,
    )
    # axis=1 means "join side by side" (add columns, not rows).
    return pd.concat([numeric_bin, cat_bin], axis=1)


X_train = build_features(train)
X_test = build_features(test)
print(f"\nBinary feature matrix: train={X_train.shape}, test={X_test.shape}")

# ---------------------------------------------------------------------
# 5. Task 2.1: k-means clustering
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("5. TASK 2.1: K-MEANS (k = 3, 5, 10)")
print("=" * 60)
# K-means is "unsupervised": it never sees the income labels. It just
# groups similar people together. How it works:
#   1. Start with k random centre points ("centroids").
#   2. Assign each person to the nearest centroid.
#   3. Move each centroid to the average of the people assigned to it.
#   4. Repeat steps 2-3 until the groups stop changing.
# Distance function: Euclidean distance (straight-line distance; this is
# scikit-learn's KMeans default).
# Because all features are 0/1, each centroid value is the FRACTION of the
# cluster's members that have that feature (e.g. 0.82 = 82% of the cluster).

for k in (3, 5, 10):
    # n_clusters=k : how many groups to find
    # n_init=10    : run the algorithm 10 times from different random
    #                starts and keep the best result
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    km.fit(X_train)    # find the clusters (no labels are used)

    # cluster_centers_ holds the centroids: one row per cluster, one column
    # per feature.
    centroids = pd.DataFrame(
        km.cluster_centers_,
        columns=X_train.columns,
        index=[f"cluster_{i}" for i in range(k)],
    ).round(3)

    # Save the full centroid table (it has ~100 columns) as a CSV file.
    path = os.path.join(OUTPUT_DIR, f"kmeans_centroids_k{k}.csv")
    centroids.to_csv(path)

    print(f"\n--- k = {k} ---")
    # km.labels_ = the cluster number assigned to each training person;
    # bincount counts how many people are in each cluster.
    print("Cluster sizes:", np.bincount(km.labels_).tolist())
    print(f"Full centroids saved to: {path}")

    # Print a compact view: each cluster's 8 strongest (highest) features,
    # which describe the "typical person" in that cluster.
    for i in range(k):
        top = centroids.loc[f"cluster_{i}"].sort_values(ascending=False).head(8)
        print(f"  cluster_{i} top features: "
              + ", ".join(f"{name}={val:.2f}" for name, val in top.items()))

# ---------------------------------------------------------------------
# 6. Task 2.2: kNN on the last 10 test records
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("6. TASK 2.2: kNN ON LAST 10 TEST RECORDS (k = 3, 5, 10)")
print("=" * 60)
# kNN (k-nearest neighbours) classifies a person by finding the k most
# similar people in the training data and letting them vote. For example,
# with k=5, if 4 of the 5 most similar people earn >50K, predict >50K.
# "Similar" is measured with Hamming distance: the fraction of 0/1 features
# on which two people differ (a good fit for binary data).

# tail(10) selects the LAST 10 rows of the test data, as the assignment asks.
X_last10 = X_test.tail(10)
y_last10 = y_test.tail(10)
knn_results = {}      # will store accuracy for each k

for k in (3, 5, 10):
    # n_neighbors=k : how many similar people get to vote
    knn = KNeighborsClassifier(n_neighbors=k, metric="hamming")
    # For kNN, fit() just stores the training data; the real work happens
    # at predict() time, when it searches for the nearest neighbours.
    knn.fit(X_train.values, y_train.values)
    pred = knn.predict(X_last10.values)
    acc = accuracy_score(y_last10, pred)
    knn_results[k] = acc

    print(f"\n--- k = {k} ---")
    print(pd.DataFrame({
        "true": y_last10.values,
        "predicted": pred,
        "correct": pred == y_last10.values,
    }).to_string())
    # With only 10 records, accuracy can only be 0.0, 0.1, ..., 1.0, so it is
    # a very rough estimate.
    print(f"Accuracy on last 10 records: {acc:.1f}")

# ---------------------------------------------------------------------
# 7. Task 3: SVM
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("7. TASK 3: SVM")
print("=" * 60)
# A Support Vector Machine looks for the best dividing boundary between the
# two classes, the one that leaves the widest possible gap on each side.
# New people are classified by which side of the boundary they fall on.

start = time.time()   # note the time so we can report how long training took
svm = SVC(
    kernel="rbf",     # allows a curved (non-straight) boundary
    C=1.0,            # strictness: higher C punishes training mistakes more
    gamma="scale",    # how far each training point's influence reaches
)
# Faster alternative if this takes too long:
#   from sklearn.svm import LinearSVC
#   svm = LinearSVC(max_iter=5000, random_state=RANDOM_STATE)
svm.fit(X_train, y_train)               # learn the boundary (may take minutes)
svm_acc = svm.score(X_test, y_test)     # score() = accuracy on the test data
print(f"SVM (RBF kernel, C=1.0) test accuracy: {svm_acc:.4f} "
      f"(trained in {time.time() - start:.0f}s)")

# ---------------------------------------------------------------------
# 8. Task 4: neural network
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("8. TASK 4: NEURAL NETWORK")
print("=" * 60)
# A neural network passes the features through layers of small calculation
# units. During training it repeatedly adjusts the weights connecting the
# units so that its mistakes shrink.

start = time.time()
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),  # two hidden layers: 64 units, then 32 units
    activation="relu",            # the function each unit applies to its sum
    max_iter=300,                 # maximum number of training passes
    random_state=RANDOM_STATE,    # repeatable starting weights
)
mlp.fit(X_train, y_train)               # train the network
mlp_acc = mlp.score(X_test, y_test)     # accuracy on the test data
print(f"Neural network (64, 32) test accuracy: {mlp_acc:.4f} "
      f"(trained in {time.time() - start:.0f}s)")

# ---------------------------------------------------------------------
# 9. Summary
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("9. SUMMARY OF ACCURACIES")
print("=" * 60)
# Note: the kNN rows are based on only 10 records, so they are not directly
# comparable with the other rows, which use the full test set.

summary = pd.DataFrame([
    ["Decision tree (Task 1, categorical only)", tree_acc_task1],
    ["Naive Bayes (Task 1, categorical only)", nb_acc_task1],
    ["kNN k=3 (last 10 test records)", knn_results[3]],
    ["kNN k=5 (last 10 test records)", knn_results[5]],
    ["kNN k=10 (last 10 test records)", knn_results[10]],
    ["SVM (Task 3)", svm_acc],
    ["Neural network (Task 4)", mlp_acc],
], columns=["model", "accuracy"])
summary["accuracy"] = summary["accuracy"].round(4)
print(summary.to_string(index=False))