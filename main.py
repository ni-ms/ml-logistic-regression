import re
import warnings

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from prettytable import PrettyTable
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

nltk.download('stopwords')
# Load the dataset
train_data = pd.read_csv('data/heading_train.csv')
train_data.head()
print("NUMBER OF DATA POINTS -", train_data.shape[0])
print("NUMBER OF FEATURES -", train_data.shape[1])
print("FEATURES -", train_data.columns.values)
train_data['Category'].value_counts()
# Check the null values
train_data.isna().sum()
target_category = train_data['Category'].unique()
print(target_category)

# Check the distribution of target variable
news_cat = train_data['Category'].value_counts()
plt.figure(figsize=(10, 6))
my_colors = ['r', 'g', 'c', 'm', 'b']
news_cat.plot(kind='bar', color=my_colors)
plt.grid()
plt.xlabel("News Categories")
plt.ylabel("Datapoints Per Category")
plt.title("Distribution of Datapoints Per Category")
plt.show()

warnings.filterwarnings("ignore")

# Load the stop words from nltk ( a, an, the, etc.)
stop_words = set(stopwords.words('english'))


def txt_preprocessing(total_text, index, column, df):
    if type(total_text) is not int:
        string = ""

        # replace_every_special_char_with_space
        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', total_text)

        # replace_multiple_spaces_with_single_space
        total_text = re.sub('\s+', ' ', total_text)

        # converting_all_the_chars_into_lower_case
        total_text = total_text.lower()

        for word in total_text.split():
            # if_the_word_is_a_not_a_stop_word_then_retain_that_word_from_the_data
            if not word in stop_words:
                string += word + " "

        df[column][index] = string


# Preprocessing the text data
for index, row in train_data.iterrows():
    if type(row['Text']) is str:
        txt_preprocessing(row['Text'], index, 'Text', train_data)
    else:
        print("THERE IS NO TEXT DESCRIPTION FOR ID :", index)

train_data.head()

# Split the data into train and cross validation
X_train = train_data
y_train = train_data['Category']

X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.20, stratify=y_train, random_state=0)

print("NUMBER OF DATA POINTS IN TRAIN DATA :", X_train.shape[0])
print("NUMBER OF DATA POINTS IN CROSS VALIDATION DATA :", X_cv.shape[0])

train_class_distribution = X_train['Category'].value_counts().sort_index()
cv_class_distribution = X_cv['Category'].value_counts().sort_index()

# distribution_of y_i's_in_train_data
plt.figure(figsize=(10, 6))
my_colors = ['r', 'g', 'b', 'k', 'y']
train_class_distribution.plot(kind='bar', color=my_colors)
plt.xlabel('CATEGORY')
plt.ylabel('DATA POINTS PER CATEGORY (CLASS)')
plt.title('DISTRIBUTION OF y_i IN TRAIN DATA')
plt.grid()
plt.show()

# -(train_class_distribution.values):_the_minus_sign_will_returns_in_decreasing_order
sorted_yi = np.argsort(-train_class_distribution.values)
for i in sorted_yi:
    print('NUMBER OF DATA POINTS IN CLASS', i + 1, ':', train_class_distribution.values[i], '(',
          np.round((train_class_distribution.values[i] / X_train.shape[0] * 100), 3), '%)')

print("-." * 50)
# distribution_of y_i's_in_cv_data
plt.figure(figsize=(10, 6))
my_colors = ['r', 'g', 'b', 'k', 'y']
cv_class_distribution.plot(kind='bar', color=my_colors)
plt.xlabel('CATEGORY')
plt.ylabel('DATA POINTS PER CATEGORY (CLASS)')
plt.title('DISTRIBUTION OF y_i IN CROSS VALIDATION DATA')
plt.grid()
plt.show()

sorted_yi = np.argsort(-cv_class_distribution.values)
for i in sorted_yi:
    print('NUMBER OF DATA POINTS IN CLASS', i + 1, ':', cv_class_distribution.values[i], '(',
          np.round((cv_class_distribution.values[i] / X_cv.shape[0] * 100), 3), '%)')

# building a CountVectorizer with all the words that occured minimum 3 times in train data
from sklearn.feature_extraction.text import CountVectorizer


# Vectorizing the text data for feature extraction
text_vectorizer = CountVectorizer(min_df=3)
train_text_ohe = text_vectorizer.fit_transform(X_train['Text'])

# getting all the feature names (words)
train_text_features = text_vectorizer.get_feature_names_out()

# train_text_ohe.sum(axis=0).A1 will sum every row and returns (1*number of features) vector
train_text_fea_counts = train_text_ohe.sum(axis=0).A1

# zip(list(text_features),text_fea_counts) will zip a word with its number of times it occured
text_fea_dict = dict(zip(list(train_text_features), train_text_fea_counts))

print("Total Number of Unique Words in Train Data :", len(train_text_features))

train_text_ohe = normalize(train_text_ohe, axis=0)

# we use the same vectorizer that was trained on train data
cv_text_ohe = text_vectorizer.transform(X_cv['Text'])

# don't forget to normalize every feature
cv_text_ohe = normalize(cv_text_ohe, axis=0)


def plot_confusion_matrix(test_y, predict_y):
    C = confusion_matrix(test_y, predict_y)  # confusion_mat
    A = (((C.T) / (C.sum(axis=1))).T)  # recall_mat
    B = (C / C.sum(axis=0))  # precision_mat

    labels = [1, 2, 3, 4, 5, 6]

    # representing_C_in_heatmap_format
    print("-" * 40, "Confusion Matrix", "-" * 40)
    plt.figure(figsize=(20, 7))
    sns.heatmap(C, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()

    # representing_B_in_heatmap_format
    print("-" * 40, "Precision Matrix (Columm Sum=1)", "-" * 40)
    plt.figure(figsize=(20, 7))
    sns.heatmap(B, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()

    # representing_A_in_heatmap_format
    print("-" * 40, "Recall Matrix (Row Sum=1)", "-" * 40)
    plt.figure(figsize=(20, 7))
    sns.heatmap(A, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()


# train a logistic regression + calibration model using text features which are one-hot encoded
alpha = [10 ** x for x in range(-5, 1)]

cv_log_error_array = []
for i in alpha:
    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)  # loss='log'_means_logistic_regression
    clf.fit(train_text_ohe, y_train)

    lr_sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    lr_sig_clf.fit(train_text_ohe, y_train)

    predict_y = lr_sig_clf.predict_proba(cv_text_ohe)
    cv_log_error_array.append(log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

    print('For values of alpha =', i, "The log loss is:", log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
fig, ax = plt.subplots()
ax.plot(alpha, cv_log_error_array, c='g')
for i, txt in enumerate(np.round(cv_log_error_array, 3)):
    ax.annotate((alpha[i], np.round(txt, 3)), (alpha[i], cv_log_error_array[i]))
plt.grid()
plt.title("Cross Validation Error for Each Alpha")
plt.xlabel("Alpha i's")
plt.ylabel("Error Measure")
plt.show()
best_alpha = np.argmin(cv_log_error_array)
clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)
clf.fit(train_text_ohe, y_train)
lr_sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
lr_sig_clf.fit(train_text_ohe, y_train)

predict_y = lr_sig_clf.predict_proba(train_text_ohe)
print('For values of best alpha =', alpha[best_alpha], "The train log loss is:",
      log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = lr_sig_clf.predict_proba(cv_text_ohe)
print('For values of best alpha =', alpha[best_alpha], "The cross validation log loss is:",
      log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

predicted_y = lr_sig_clf.predict(cv_text_ohe)
train_accuracy = (lr_sig_clf.score(train_text_ohe, y_train) * 100)
cv_accuracy = (accuracy_score(predicted_y, y_cv) * 100)

print("Logistic Regression Train Accuracy -", train_accuracy)
print("Logistic Regression CV Accuracy -", cv_accuracy)

plot_confusion_matrix(y_cv, lr_sig_clf.predict(cv_text_ohe.toarray()))

print(classification_report(predicted_y, y_cv, target_names=target_category))

x = PrettyTable()
x.field_names = ["Model", "Train Log-Loss", "CV Log-Loss", "Train Accuracy", "CV Accuracy"]
x.add_row(['Logistic Regression', '0.034', '0.132', '100.0', '96.64'])
print(x)
test_data = pd.read_csv("data/heading_test.csv")
test_data.shape
test_data.head()
# checking null values
test_data.isna().sum()

# test_data_text_processing_stage_
for index, row in test_data.iterrows():
    if type(row['Text']) is str:
        txt_preprocessing(row['Text'], index, 'Text', test_data)
    else:
        print("THERE IS NO TEXT DESCRIPTION FOR ID :", index)

test_data.head()
# we use the same vectorizer that was trained on train data
test_text_ohe = text_vectorizer.transform(test_data['Text'])

# don't forget to normalize every feature
test_text_ohe = normalize(test_text_ohe, axis=0)

# lr_sig_clf is the same CalibratedClassifierCV which is used in Logistic Regression Model
test_final_ohe = lr_sig_clf.predict(test_text_ohe)
test_final_list = test_final_ohe.tolist()
test_final_list[:5]
test_data['Category'] = test_final_list
test_data.head(20)
test_data = test_data.drop("Text", axis=1)
test_data.head(20)
