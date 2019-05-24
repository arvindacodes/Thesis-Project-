"""
Author: Arvind Ramesh
Reg No: R00171371
Msc. Artificial Intelligence

"""
import nltk
import warnings
warnings.filterwarnings("ignore")
import operator
# nltk.download() # for downloading packages
import json
import numpy as np
import random
import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import time
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
##########Sentiment Model####################



"""Model 1: Sentiment analysis model using Bernoulli's Naive bayes referenced and inspired from Machine learning Notes and github"""
##########Sentiment Model####################

"""opening of NRC lexicon"""
nrcDF_lexicon = pd.read_csv("NRC-Emotion-Lexicon-Wordlevel-v0.92.txt", names=["word", "emotion", "association"], sep='\t')
reshap_lexicon = nrcDF_lexicon.pivot(index='word', columns='emotion', values='association').reset_index()
# print(nrcDF_lexicon['emotion'].unique())
def Input_data():
    root = "Datasets/"

    with open(root + "imdb_labelled.txt", "r") as text_file:
        Input = text_file.read().split('\n')

    with open(root + "amazon_cells_labelled.txt", "r") as text_file:
        Input += text_file.read().split('\n')

    with open(root + "yelp_labelled.txt", "r") as text_file:
        Input += text_file.read().split('\n')

    return Input
values = Input_data()

def Preprocess_data(Input):
    Processed_data = []
    for single in Input:
        if len(single.split("\t")) == 2 and single.split("\t")[1] != "":
            Processed_data.append(single.split("\t"))

    return Processed_data

all_data = Input_data()
values = Preprocess_data(all_data)

def seperate_data(Input):
    total_length = len(Input)
    training_ratio = 0.75
    train_data = []
    evaluate_data = []

    for indices in range(0, total_length):
        if indices < total_length * training_ratio:
            train_data.append(Input[indices])
        else:
            evaluate_data.append(Input[indices])

    return train_data, evaluate_data

def preprocessing_step():
    Input = Input_data()
    Processed_data = Preprocess_data(Input)

    return seperate_data(Processed_data)

def training_step(Input, vectorizer):
    train_text = [Input[0] for Input in Input]
    train_result = [Input[1] for Input in Input]

    train_text = vectorizer.fit_transform(train_text)

    return BernoulliNB().fit(train_text, train_result)

train_data, evaluate_data = preprocessing_step()
vectorizer = CountVectorizer(binary = 'true')
classifier = training_step(train_data, vectorizer)
result = classifier.predict(vectorizer.transform(["hi hello how are u"]))

result[0]

def analyse_text(classifier, vectorizer, text):
    return text, classifier.predict(vectorizer.transform([text]))


def print_result(result):
    neagtive = 0
    positive = 0
    text, analysis_result = result
    # print("result is:", result)
    print_text = "Positive" if analysis_result[0] == '1' else "Negative"
    # print(text,analysis_result)
    print(text, ":", print_text)
    return print_text

def simple_evaluation(evaluate_data):
    evaluate_text     = [evaluate_data[0] for evaluate_data in evaluate_data]
    evaluate_result   = [evaluate_data[1] for evaluate_data in evaluate_data]

    total = len(evaluate_text)
    corrects = 0
    for index in range(0, total):
        analysis_result = analyse_text(classifier, vectorizer, evaluate_text[index])
        text, result = analysis_result
        corrects += 1 if result[0] == evaluate_result[index] else 0

    return corrects * 100 / total

# print_result(new_result)

#Model 2 : Chat-Bot
"""The coding of this Chatbot model was inspired from Author Parul Pandey/https://github.com/parulnith"""
"""opening of the test file """

f=open('text.txt','r',errors = 'ignore')

input_file=f.read()
rawInputfile=input_file.lower()# converts to lowercase
# nltk.download('punkt') # first-time use only
# nltk.download('wordnet') # first-time use only
sent_tokens = nltk.sent_tokenize(input_file)# convhierts to list of sentences
word_tokens = nltk.word_tokenize(input_file)# converts to list of words

lemmer = nltk.stem.WordNetLemmatizer()


def lemmetize_tokens(words):
    return [lemmer.lemmatize(word) for word in words]
punc_removal = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return lemmetize_tokens(nltk.word_tokenize(text.lower().translate(punc_removal)))

input_greet = ("hello", "hi", "greetings", "sup", "what's up","hey",)
greet_respond = ["hello", "hi", "greetings", "sup", "what's up","hey",'*nods*' ]



# Checking for greetings
def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for i in sentence.split():
        if i.lower() in input_greet:
            return random.choice(greet_respond)
# Generating response
def response(user_response):
    Arvbot_response=''
    sent_tokens.append(user_response)
    calculate_tfidf = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    total_tfidf = calculate_tfidf.fit_transform(sent_tokens)
    vals = cosine_similarity(total_tfidf[-1], total_tfidf)
    idx=vals.argsort()[0][-2]
    # print(idx)

    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        Arvbot_response=Arvbot_response+"I am sorry! I don't understand you"
        return Arvbot_response
    else:
        Arvbot_response = Arvbot_response+sent_tokens[idx]
        return Arvbot_response
""""""
def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    print("{:-<40} Here{}".format(sentence, str(score)))
    return score

flag=True
print("Arvbot: My name is Arvbot. I will answer your queries about movies. If you want to exit, type Bye!")

"""Model 3: Lexicon Based emotion analysis,with reference to NLP class notes and slides"""
# def input():
while(flag==True):
    sad = 0  # completed
    joy = 0  # completed
    disgust = 0  # completed
    fear = 0  # completed
    surprise = 0  # completed
    anger = 0  # completed
    anticipation = 0  # completed
    trust = 0  # completed
    others = 0
    user_response = input()
    user_response=user_response.lower()
    checkword = user_response.split()
    # print("splitted word is ", checkword)
    for words in range(len(checkword)):
        if len(reshap_lexicon[(reshap_lexicon.word == checkword[words]) & (reshap_lexicon.joy == 1)].values) != 0:
            joy += 1
        elif len(reshap_lexicon[(reshap_lexicon.word == checkword[words]) & (reshap_lexicon.sadness == 1)].values) != 0:
            sad += 1
        elif len(reshap_lexicon[(reshap_lexicon.word == checkword[words]) & (reshap_lexicon.anger == 1)].values) != 0:
            anger += 1
        elif len(reshap_lexicon[(reshap_lexicon.word == checkword[words]) & (reshap_lexicon.fear == 1)].values) != 0:
            fear += 1
        elif len(reshap_lexicon[(reshap_lexicon.word == checkword[words]) & (reshap_lexicon.disgust == 1)].values) != 0:
            disgust += 1
        elif len(
                reshap_lexicon[(reshap_lexicon.word == checkword[words]) & (reshap_lexicon.surprise == 1)].values) != 0:
            surprise += 1
        elif len(reshap_lexicon[
                     (reshap_lexicon.word == checkword[words]) & (reshap_lexicon.anticipation == 1)].values) != 0:
            anticipation += 1
        elif len(reshap_lexicon[(reshap_lexicon.word == checkword[words]) & (reshap_lexicon.trust == 1)].values) != 0:
            trust += 1
        elif 1 in (reshap_lexicon[
            (reshap_lexicon.word == checkword[words]) & (reshap_lexicon.anger == 0) & (reshap_lexicon.joy == 0) &
            (reshap_lexicon.fear == 0) & (reshap_lexicon.disgust == 0) & (reshap_lexicon.surprise == 0) & (
                    reshap_lexicon.anticipation == 0)
            & (reshap_lexicon.trust == 0) & (reshap_lexicon.sadness == 0)].values):
            # increment the other variable by 1
            others += 1
    # define a dic to store the count of emotion from lexicon
    countemodict = {"Joy": joy,"Angry": anger, "Sad": sad, "Trust": trust,  "Anticipation": anticipation,
                    "Fear": fear, "disgust": disgust, "surprise": surprise, "others": others}
    print("dict is ", countemodict)

    # the max count of the emotion is stored in the label_col filed
    total_count = max(countemodict.items(), key=operator.itemgetter(1))[0]
    print("total emotion count is ", total_count)

    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("Arvbot: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("Arvbot: "+greeting(user_response))
            else:
                sentiment = analyse_text(classifier, vectorizer, user_response)
                analyser = SentimentIntensityAnalyzer()
                score=sentiment_analyzer_scores(user_response)
                # print("score is ",score)

                emotion=(print_result(sentiment).split())
                L = emotion
                # print("l check is ",L)
                makeitastring = ''.join(map(str, L))
                # print("str check is ",makeitastring)

                # print("this is the checkpoint",str(emotion))
                sadness = 0
                joyness = 0
                if (total_count == 'Joy') & (makeitastring =='Negative'):
                    # sadness += 1
                    final_emotion = 'sad'
                    print("final emotion is:", final_emotion)
                elif (total_count == 'Sad') &(makeitastring=='Positive'):
                    final_emotion = 'joy'
                elif (total_count == 'Trust')& (makeitastring =='Negative'):
                    final_emotion == 'Angry'
                    # joyness
                    print("final emotion is:", final_emotion)
                else: final_emotion = total_count
                print("the final emotion is:",final_emotion)
                if final_emotion == 'Angry':
                    p = 'Relax and breath'
                elif final_emotion == 'Sad':
                    p = 'Cheer up and smile'
                elif final_emotion == 'sad':
                    p = 'Cheer up and smile'
                elif final_emotion == 'surprise':
                    p= "its good to be surprised"
                elif final_emotion == 'Fear':
                    p = "Common you are grown up"
                elif final_emotion == 'disgust':
                    p = "Let it go and move on "
                elif final_emotion == 'others':
                    p = "Hmm what is going on in your mind"
                else:
                    p = "Hurray"
                take_sentence = p
                print("Arvbot: ", end="")
                # print(take_sentence)
                chatresponse = response(user_response)
                ab=chatresponse.lower()
                sentiment1 = analyse_text(classifier, vectorizer, ab)
                sent_tokens.remove(user_response)

                # a= (print_result(sentiment1))
                # print("a is :",a)
                # print("sentiment1 is ", sentiment1[0])
                b= sentiment1[0]
                c = take_sentence+' '+ b
                sentiment_final = analyse_text(classifier,vectorizer,c)
                print_result(sentiment_final)
                # ab = sentiment_final
                print("here",ab)
                analyser = SentimentIntensityAnalyzer()
                score1 = sentiment_analyzer_scores(c.lower())
                print(score1)
                checkword_response = str(sentiment_final).split()
                # print("splitted word is ", checkword)
                for words in range(len(checkword_response)):
                    if len(reshap_lexicon[
                               (reshap_lexicon.word == checkword_response[words]) & (reshap_lexicon.joy == 1)].values) != 0:
                        joy += 1
                    elif len(reshap_lexicon[(reshap_lexicon.word == checkword_response[words]) & (
                            reshap_lexicon.sadness == 1)].values) != 0:
                        sad += 1
                    elif len(reshap_lexicon[
                                 (reshap_lexicon.word == checkword_response[words]) & (reshap_lexicon.anger == 1)].values) != 0:
                        anger += 1
                    elif len(reshap_lexicon[
                                 (reshap_lexicon.word == checkword_response[words]) & (reshap_lexicon.fear == 1)].values) != 0:
                        fear += 1
                    elif len(reshap_lexicon[(reshap_lexicon.word == checkword_response[words]) & (
                            reshap_lexicon.disgust == 1)].values) != 0:
                        disgust += 1
                    elif len(
                            reshap_lexicon[(reshap_lexicon.word == checkword_response[words]) & (
                                    reshap_lexicon.surprise == 1)].values) != 0:
                        surprise += 1
                    elif len(reshap_lexicon[
                                 (reshap_lexicon.word == checkword_response[words]) & (
                                         reshap_lexicon.anticipation == 1)].values) != 0:
                        anticipation += 1
                    elif len(reshap_lexicon[
                                 (reshap_lexicon.word == checkword_response[words]) & (reshap_lexicon.trust == 1)].values) != 0:
                        trust += 1
                    elif 1 in (reshap_lexicon[
                        (reshap_lexicon.word == checkword_response[words]) & (reshap_lexicon.anger == 0) & (
                                reshap_lexicon.joy == 0) &
                        (reshap_lexicon.fear == 0) & (reshap_lexicon.disgust == 0) & (reshap_lexicon.surprise == 0) & (
                                reshap_lexicon.anticipation == 0)
                        & (reshap_lexicon.trust == 0) & (reshap_lexicon.sadness == 0)].values):
                        # increment the other variable by 1
                        others += 1
                # define a dic to store the count of emotion from lexicon
                countemodict1 = {"Joy": joy, "Angry": anger, "Sad": sad, "Trust": trust, "Anticipation": anticipation,
                                "Fear": fear, "disgust": disgust, "surprise": surprise, "others": others}
                print("dict is ", countemodict1)

                # the max count of the emotion is stored in the label_col filed
                total_count1 = max(countemodict1.items(), key=operator.itemgetter(1))[0]
                print("total emotion count is ", total_count1)
                l1 = emotion=(print_result(sentiment_final).split())
                makeitastring1 = ''.join(map(str, l1))
                # print("str check here is ", makeitastring1)
                if (total_count1 == 'Fear') & (makeitastring1 =='Positive'):
                    # sadness += 1
                    final_emotion1 = 'Joy'
                    print("final emotion is:", final_emotion1)
                elif (total_count1 == 'Sad') &(makeitastring1=='Positive'):
                    final_emotion1 = 'joy'
                elif (total_count1 == 'Joy')& (makeitastring1 =='Negative'):
                    final_emotion1 = 'Joy'
                    # joyness
                    print("final emotion is:", final_emotion1)
                else: final_emotion1 = total_count1
                print("the final emotion is:",final_emotion1)


                with open('output.txt', 'a') as the_file:
                    the_file.write(user_response+'\t'+total_count+'\t'+makeitastring+'\t'+final_emotion+'\n'+json.dumps(score)+'\n'+
                                   chatresponse+'\t'+total_count1+'\t'+makeitastring1+'\t'+final_emotion1+'\n'+json.dumps(score1)+'\n'*2)


else:
    flag=False
    print("Arvbot: Bye! take care..")
def create_confusion_matrix(evaluate_data):
    evaluate_text = [evaluate_data[0] for evaluate_data in evaluate_data]
    actual_result = [evaluate_data[1] for evaluate_data in evaluate_data]
    prediction_result = []
    for text in evaluate_text:
        analysis_result = analyse_text(classifier, vectorizer, text)
        prediction_result.append(analysis_result[1][0])

    matrix = confusion_matrix(actual_result, prediction_result)
    return matrix

import pandas as pd
confusion_matrix = create_confusion_matrix(evaluate_data)
pd.DataFrame(confusion_matrix, columns=["Negatives", "Positives"],index=["Negatives", "Positives"])

classes = ["Negatives", "Positives"]

plt.figure()
plt.imshow(confusion_matrix, interpolation='nearest')
plt.title("Confusion Matrix for Sentiment Analysis on the dataset")
# plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=30)
plt.yticks(tick_marks, classes)

text_format = 'd'
thresh = confusion_matrix.max() / 2.
for row, column in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
    plt.text(column, row, format(confusion_matrix[row, column], text_format),
             horizontalalignment="center",
             color="white" if confusion_matrix[row, column] > thresh else "black")

plt.ylabel('Correct label')
plt.xlabel('label predicted')
plt.tight_layout()

plt.show()
TN = confusion_matrix[0][0] #calculating True Negative
TP = confusion_matrix[1][1] #calculating True Positive
FN = confusion_matrix[0][1] #calculating False Negative
FP = confusion_matrix[1][0] #calculating False Positive


accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2*(recall * precision) / (recall + precision)

print('Accuracy of Sentiment model is  :',accuracy)
print('Precision of Sentiment model is  :',precision)
print('Recall of Sentiment model is :',recall)
print('F1 Score of Sentiment model is:',f1_score)
