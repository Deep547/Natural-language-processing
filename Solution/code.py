# nltk.download()
import nltk

nltk.download('punkt')
import json
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import bigrams
from nltk.tokenize import word_tokenize


def search(string,
           desired_result):  # The purpose of this function is to search positive and negative words in the corpus using binary search.
    begin = 0  # begin variable is initialized to 0.
    finish = len(
        string) - 1  # its value is set to one less than the length of the entire "string".The string argument in our case will be positive_word and negative_words.
    while begin <= finish:  # while loop is applied with condition that value of begin variable is less than finish.
        centre = int((
                                 begin + finish) / 2)  # we identify the central location(of positive words corpus) for search by considering the integral average of begin and finish.
        focus = string[centre]  # then the string at the central location is assigned to the variable focus.
        if focus > desired_result:  # we compare the values(strings) of focus and desired_result.if length of focus string is greater than desired result
            finish = centre - 1  # new value of the finish variable becomes center reduced by 1
        elif focus < desired_result:  # however,if the length of focus variable is lesser than desired result then, begin variable becomes equivalent to centre+1.
            begin = centre + 1
        else:  # if none of the above condition holds i.e both variables are of same size then focus is returned
            return focus  #
    return -1  # once the condition begin <= finish no more holds true the loop quits


def find_word(
        token):  # The purpose of this function is to determine the number of positive and negative words in the corpus provided
    negative = 0  # variables negative which implies number of negative words is initialized to value 0.
    positive = 0  # variable positive which implies the number of positive words is initialized to value 0.
    for ind in token:  # for loop to loop through the lemmatized tokens stored in the token list
        i = search(positive_word,
                   ind)  # search function is called and positive_word and ind variables are sent as arguments.The output returned is stored in the variable i.
        if (
                i != -1):  # if the value assigned to i variable is not -1 then,count of the variable positive is incremented by 1
            positive += 1
        else:  # However when i has value -1 assigned to it
            i = search(negative_word,
                       ind)  # search function is called and positive_word and ind variables are sent as arguments.The output returned is stored in the variable i.
            if (i != -1):  # again if the value returned is not -1 then the negative variable is incremented by 1
                negative += 1
    return positive, negative  # finally the function returns the value of the positive and the negative variables.


lemma = WordNetLemmatizer()  # the wordnet lemmatizer is being used over here
news_variety, training, testing, token, negative_word, positive_word = [], [], [], [], [], []  # defining a set of empty lists
news_positive, news_negative, counter = 0, 0, 0  # initializing the variables with zero value
with open(
        "signal-news1.jsonl") as file:  # opening the jsonl file,purpose here is to pre process file as per the conditions required
    for line in file:  # looping through the lines of the file
        lines = json.loads(line)
        content = (lines['content'])  # reading the content section
        content = content.lower()  # reducing entire content to lower case
        a = re.sub(r'https?:\/\/\S+[\r\n]*|(www\.|reut\.rs|bit\.ly)\S+|\S+\.(com|org|uk|info)', '', content,
                   flags=re.MULTILINE)  # removing the various types of URL's
        a = re.sub(r'(?:(?:http?|https?):\/\/)?([-a-zA-Z0-9.]*\.[a-z]*)\b(?:\/[-a-zA-Z0-9@:%_\+.~?&//=]*)?', '',
                   content)  # removing url's while checking some added constraints.
        b = re.sub(r'[^a-zA-Z0-9\s]', '', a)  # remove all the non-alphanumeric characters except spaces
        c = re.sub(r'\b\w{1,3}\b', '', b)  # Remove words with 3 characters or fewer
        d = re.sub(r'\b[0-9]+\b', '', c)  # removing numbers fully made up of digits
        a = word_tokenize(
            d)  # tokens are generated for the formatted text received(in d) from the regular expression done above, and assigned to variable a.
        token += (lemma.lemmatize(e) for e in
                  a)  # all the tokens in a are lemmatized using a for loop and the output is assigned to the variable token
        news_variety.append(
            d)  # The preprocessed output obtained by filtering the contents of json file after applying the regular expression is stored into the list news_variety
        if counter < 16000:  # if condition checks if counter is less than 16000 and if the condition is true, it lemmatizes all the tokens in a assigning these tokens to list training
            training += (lemma.lemmatize(e) for e in a)
        else:  # when the number of lines becomes greater than 16000,the lemmatized tokens obtained from a are assigned to the list testing.
            testing += (lemma.lemmatize(e) for e in a)
        counter += 1  # counter is incremented at the end of for loop

print("The Total number of tokens is : " + str(len(token)))  # prints the total number of tokens.
print("The Total vocabulary size is : " + str(len(set(token))))  # prints the total vocabulary size

dict = {}  # empty dictionary is declared here in order to store the bigrams into it and to access them quickly
for twogram in bigrams(
        token):  # for loop to loop through bigrams of token variable.Using the else part the first bigram is inserted into the dictionary dict
    if twogram in dict:  # and then the for loop uses the if loop to update the members inside the dictionary dict.
        dict[twogram] += 1
    else:
        dict[twogram] = 1
# print(dict)

frequent_bigrams = sorted(dict, key=dict.get,
                          reverse=True)  # accessing the dictionary of bigrams ordered by highest frequency
print("The most frequent bigrams are:\n",
      frequent_bigrams[:25])  # lists the top 25 bigrams based on the number of occurrences on the entire corpus.

with open("negative-words.txt",
          encoding="ISO-8859-1") as a:  # opening the file negative-words.txt as a to read the contents by specifying the encoding
    file = a.readlines()[
           35:]  # reading the lines from file a, after the line 35 into the variable file(as lines from 1-35 can be ignored)
    for lines in file:  # using a for loop to loop through each line in the file
        negative_word.append(
            lines.strip())  # negative_word list here is appended after stripping each of the lines from the default whitespace characters.

with open("positive-words.txt",
          encoding="ISO-8859-1") as a:  # opening the file positive-words.txt as a to read the contents by specifying the encoding
    file = a.readlines()[
           35:]  # reading the lines from file a, after the line 35 into the variable file(as lines from 1-35 can be ignored)
    for lines in file:  # using a for loop to loop through each line in the file
        positive_word.append(
            lines.strip())  # positive_word list here is appended after stripping each of the lines from the default whitespace characters.

pos, neg = find_word(
    token)  # both the values returned by the function find_word(),after passing the lemmatized token as argument are assigned to the variables pos and neg.

print("The count of the positive words is : " + str(pos))  # printing the number of positive words
print("The count of the negative words is : " + str(neg))  # printing the number of negative words

for news in news_variety:  # for loop to find news story with more positive than negative words and with more negative than positive words.
    token = news.split()  # splits every line into strings and assigns it to the variable token
    pos, neg = find_word(
        token)  # find_word function is called by passing token variable as argument and output returned is assigned to the variables pos and neg
    if pos > neg:  # if pos variable is greater than neg then news_positive variable is incremented by 1,which means news is considered more positive
        news_positive += 1
    else:  # However, in other situation news_negative variable is incremented by 1,which means news is considered more negative
        news_negative += 1
print("The number of Positive news stories is : " + str(
    news_positive))  # prints the total number of positive news stories
print("The number of Negative news stories is : " + str(
    news_negative))  # prints the total number of negative news stories

training_pair = {} #An empty dictionary training_pair is created
for twogram in bigrams(training): #for loop to loop through bigrams of training variable.Using the else part the first bigram is inserted into the dictionary testing_pair
    if twogram in training_pair:
        training_pair[twogram] += 1 #the for loop uses the if loop to update the members inside the dictionary testing_pair.
    else:
        training_pair[twogram] = 1
training_frequent_bigrams = sorted(training_pair, key=training_pair.get, reverse=True) # accessing the dictionary(training_pair) of bigrams ordered by highest frequency
#print(training_frequent_bigrams)
sentence_they = ["they"] # variable declared
for i in range(0, 9): # for loop loops through values 0 to 9
    for a in training_frequent_bigrams: # nested for loop to loop through the members of training_frequent_bigrams dictionary
        if a[0] == sentence_they[i]: # checks continuously if the first word of the sentence is they
            break
    sentence_they.append(a[1]) # keeps on appending the sentence when the if loop breaks.
print('The sentence predicted to start with "They" of length 10 words is:\n'," ".join(sentence_they))

testing_pair = {}  # An empty dictionary testing_pair is created
for twogram in bigrams(
        testing):  # for loop to loop through bigrams of testing variable.Using the else part the first bigram is inserted into the dictionary testing_pair
    if twogram in testing_pair:
        testing_pair[
            twogram] += 1  # the for loop uses the if loop to update the members inside the dictionary testing_pair.
    else:
        testing_pair[twogram] = 1
        testing_frequent_bigrams = sorted(testing_pair, key=testing_pair.get,reverse=True)  # accessing the dictionary(testing_pair) of bigrams ordered by highest frequency
#print(testing)
