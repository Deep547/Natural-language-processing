import nltk
import pandas as pd
import numpy as np
import csv
import re
import gensim
from nltk import word_tokenize
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from nltk.classify.util import apply_features
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score,classification_report
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
stop_words = stopwords.words('english') # access the stop words from nltk and cease analysis when you find words in tweet like is, a, uff, hmmm, uhh
white_list_words=['no','we','have','not'] # list of the white listed words
for ind in stop_words: # for loop to remove the whitelisted words from nltk based stop_words list, if it contains them
    if ind in white_list_words:
        stop_words.remove(ind)
def repeat_char_repl(tweets): # function to replace the words in tweets with repeating characters
    return re.sub(r'(.)\1+', r'\1\1',tweets)
def token(tweets):
    tokenz = word_tokenize(tweets) #using the nltk package to tokenize the words in tweets
    tokenized = ''
    for j in tokenz:
        tokenized += str(j +' ') # adding all the tokens with space to the tokenized variable
    return tokenized.strip() # returns the tokenized variable after stripping the spaces
def special_replace(tweets): # function to replace the special characters from tweet
    tweets=re.sub(r'\#\@\S*','',tweets)
    return tweets
def non_alpha_num_replace(tweets): # function to remove non alphanumeric characters from tweets
    tweets=re.sub(r'[^a-zA-Z0-9\s]', '',tweets)
    return tweets
def num_replace(tweets): # function to remove numbers completely made of digits
    tweets=re.sub(r'\b[0-9]*\b', '' , tweets)
    return tweets
def space_replace(tweets): # function to remove the extra spaces from the tweets
    tweets=re.sub(' +',' ',tweets)
    return(tweets)
def URL_replace(tweets): # function to remove the url's from the tweets
    tweets=re.sub(r'\bhttps?:\/\/\S+[\r\n]*|(www\.|reut\.rs|bit\.ly)\S+|\S+\.(com|org|uk|info)\b','',tweets)
    tweets = re.sub(r'(?:(?:http?|https?):\/\/)?([-a-zA-Z0-9.]*\.[a-z]*)\b(?:\/[-a-zA-Z0-9@:%_\+.~?&//=]*)?', '',tweets)
    return tweets
def preprocessing(tweets): # This function calls all the above preprocessing functions and returns the processed tweets
    tweets = tweets.lower()
    tweets=URL_replace(tweets)
    tweets = special_replace(tweets)
    tweets=repeat_char_repl(tweets)
    tweets=non_alpha_num_replace(tweets)
    tweets=num_replace(tweets)
    tweets=space_replace(tweets)
    tweets=token(tweets)
    return tweets
list=[]
with open('twitter-training-data.txt',encoding="utf-8") as file:
    row=file.readlines()
    for x in row:
        x=x.strip() #stripping default whitespace characters from each of the lines of file.
        term = x.split('\t') #Splits the line on one or more tabs/spaces
        list.append(term) #appends the values in term to the list
list=pd.DataFrame(list) #converting the list into pandas dataframe
list.columns=names=['tweet_id','sentiment','tweet_text'] #assigning headers to the specific columns in the list
list.loc[:,"tweet_text"] = list.tweet_text.apply(lambda x: preprocessing(x))#using panda to access and preprocess the column tweet_text.
list.loc[:,"token_text"] = list.tweet_text.apply(lambda x: [ind for ind in x.split() if ind not in (stop_words)]) #removing the stop words form the preprocessed twitter training data and assigning it to tweet_text
list.loc[:,"tweet_text"] = list.token_text.apply(lambda x: " ".join(x)) #joins the words under the header token_text with space and assigns it to dataframe header tweet_text
#The purpose of the pipeline is to assemble several steps that can be cross-validated together while setting different parameters
#CountVectorizer from NLTK is used as vectizer to Convert a collection of text documents to a matrix of token counts
#min_df=3 means When building the vocabulary ignore terms that have a document frequency strictly lower than 3.
#ngram_range specifies the lower(here one grams) and upper boundary(here 2 grams) of the range of n-values for different n-grams to be extracted.
#Logistic regression has been used as the classifier
#random_state is the random number generator
pipe = Pipeline([('vectorizer',CountVectorizer(min_df=3, ngram_range=(1,2))),('classifier',LogisticRegression(random_state=0) )])
pipe.fit(list['tweet_text'].values, list['sentiment'].values) # fitting the logistic regression model with the list of tweet_text and sentiments
development=[] # empty list
with open('twitter-test3.txt',encoding="utf-8") as file: # opening the twitter test dataset
    row=file.readlines()
    for x in row: # loop to fetch the lines in the file and to strip them of white space characters and further split the lines on tab spaces and append the values in a list.
        x=x.strip()
        term = x.split('\t')
        development.append(term)
testing,list_t,identity=[],[],[] # three different lists defined.
for i in range(len(development)): # for loop to loop through all the values in the list development
    development[i][2]=preprocessing(development[i][2]) #looping through the list that benhaves like a two dimensional array, development, to access the text section of the tweets and to preprocess it.
    testing.append(development[i][2]) #every preprocessed tweet text from the development list is appended into the testing list
    list_t.append(development[i][1]) # every sentiment from the development list is appended into the list_t
    identity.append(development[i][0]) # every tweet id from the development list is appended into the list identity
development=pd.DataFrame(development) #converting the development list into pandas dataframe
development.columns=names=['tweet_id','sentiment','tweet_text'] # assigning the column names to the dataframe development
development.loc[:,"token_text"] = list.tweet_text.apply(lambda x: [ind for ind in x.split() if ind not in (stop_words)]) #removing the stop words form the tweet_text and assigning it to token_text
prediction=pipe.predict(testing)#Predict using the linear regression model for the list testing that contains the preproceesed tweets text only with stop words removed , from file twitter-test3.txt
print(prediction[300:350])
accuracy_score(list_t,prediction)# Accuracy score predicts the accuracy of the correct predictions. Here list_t stores the true values of twitter sentiments and prediction holds the predicted values of twitter sentiments
print(classification_report(list_t, prediction, target_names=set(list['sentiment'].values))) # classification report shows the main classification metrics.It takes the lists list_t,prediction as input and target names is set to the values of sentiments
accuracy_score(prediction,list_t)
pipe_line = Pipeline([('vectorizer',CountVectorizer(min_df=3, ngram_range=(1,2))),('tfidf', TfidfTransformer(norm='l2', smooth_idf=True, sublinear_tf=False)),("classifier", LogisticRegression(random_state=0))])
pipe_line.fit(list['tweet_text'].values, list['sentiment'].values)
prediction=pipe_line.predict(testing)
print(prediction[300:350])
accuracy_score(prediction,list_t)
print(classification_report( prediction,list_t, target_names=set(list['sentiment'].values)))
pipe_line = Pipeline([('vectorizer',CountVectorizer(min_df=3, ngram_range=(1,2))),('tfidf', TfidfTransformer(norm='l2', smooth_idf=True, sublinear_tf=False)),("classifier", MultinomialNB())])
pipe_line.fit(list['tweet_text'].values, list['sentiment'].values)
prediction=pipe_line.predict(testing)
print(prediction[300:350])
accuracy_score(prediction,list_t)
print(classification_report( prediction,list_t, target_names=set(list['sentiment'].values)))