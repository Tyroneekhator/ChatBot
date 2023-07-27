import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer=LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

#opening a json file
with open("intents.json") as file:
    data=json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words=[]
    labels=[]
    docs_x=[]#list of all the different patterns
    docs_y=[]#corresponding enteries for every response for each pattern

    #code below for stemming
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)#tokenizing the words
            words.extend(wrds)
            docs_x.append(wrds)#appending tokenized words
            docs_y.append(intent["tag"])
    #to get our words
        if intent["tag"] not in labels:#puts the tages in the label list
            labels.append(intent["tag"])
    #stemming the word list below
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))#removing duplicate elements and convert it to  a list.

    labels = sorted(labels)#arranges the labels list
        #BAG OF WORDS
    training =[]
    output=[]#it contains all of the tags

    out_empty = [0 for _ in range(len(labels))]
    # creating a  bag of words

    for x,doc in enumerate(docs_x):
        bag=[]

        wrds = [stemmer.stem(w) for w in doc]#stemming the patterns

        for w in words:#looping through the words list
            if w in wrds:#if the word is found put 1 into the list
                bag.append(1)
            else:
                bag.append(0)

        output_row= out_empty[:]
        output_row[labels.index(docs_y[x])] = 1# allows the machine to add 1 to output row idf the label is found

        training.append(bag)
        output.append(output_row)

    with open("data.pickle", "wb") as f:
          pickle.dump((words, labels, training, output),f)#write all these variables into a pickle file

# changing lists into arrays
    training = numpy.array(training)
    output= numpy.array(output)

#The AI aspect of the cold
tensorflow.reset_default_graph()

net = tflearn.input_data(shape =[None, len(training[0])])
net = tflearn.fully_connected(net, 10)#eight nuerons for the hidden layer
net = tflearn.fully_connected(net, 10)#hidden layer of 8 neurons
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")#allows to get probability for each output.
net = tflearn.regression(net)

model = tflearn.DNN(net)#type of neuron network

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=6000 , batch_size=10 , show_metric=True)
    model.save("model.tflearn")#save the model

def bag_of_words(s, words):
    bag=[0 for _ in range(len(words))]

    s_words= nltk.word_tokenize(s)#list of tokenized words
    s_words = [stemmer.stem(word.lower()) for word in s_words]#stem tokenized words

    for se in s_words:
        for i,w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)#turn bag into an array

def chat():
    print("start talking with the bot(type quit to stop)!")
    while True:
        inp = input("you: ")
        if inp.lower() == "bye" :
            break
        elif inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)#helps to pick the greatest probability
        tag = labels[results_index]

        if results[results_index] > 0.8:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            print(random.choice(responses))
        else:
            print(" I did not get that ,please ask another question")


chat()
