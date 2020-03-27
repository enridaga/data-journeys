
# read in some helpful libraries

import nltk # the natural langauage toolkit, open-source NLP

import pandas as pd # dataframes



### Read in the data



# read our data into a dataframe

texts = pd.read_csv("../input/train.csv")



# look at the first few rows

texts.head()
### Split data



# split the data by author

byAuthor = texts.groupby("author")



### Tokenize (split into individual words) our text



# word frequency by author

wordFreqByAuthor = nltk.probability.ConditionalFreqDist()



# for each author...

for name, group in byAuthor:

    # get all of the sentences they wrote and collapse them into a

    # single long string

    sentences = group['text'].str.cat(sep = ' ')

    

    # convert everything to lower case (so "The" and "the" get counted as 

    # the same word rather than two different words)

    sentences = sentences.lower()

    

    # split the text into individual tokens    

    tokens = nltk.tokenize.word_tokenize(sentences)

    

    # calculate the frequency of each token

    frequency = nltk.FreqDist(tokens)



    # add the frequencies for each author to our dictionary

    wordFreqByAuthor[name] = (frequency)

    

# now we have an dictionary where each entry is the frequency distrobution

# of words for a specific author.     
# see how often each author says "blood"

for i in wordFreqByAuthor.keys():

    print("blood: " + i)

    print(wordFreqByAuthor[i].freq('blood'))



# print a blank line

print()



# see how often each author says "scream"

for i in wordFreqByAuthor.keys():

    print("scream: " + i)

    print(wordFreqByAuthor[i].freq('scream'))

    

# print a blank line

print()



# see how often each author says "fear"

for i in wordFreqByAuthor.keys():

    print("fear: " + i)

    print(wordFreqByAuthor[i].freq('fear'))
# One way to guess authorship is to use the joint probabilty that each 

# author used each word in a given sentence.



# first, let's start with a test sentence

testSentence = "It was a dark and stormy night."



# and then lowercase & tokenize our test sentence

preProcessedTestSentence = nltk.tokenize.word_tokenize(testSentence.lower())



# create an empy dataframe to put our output in

testProbailities = pd.DataFrame(columns = ['author','word','probability'])



# For each author...

for i in wordFreqByAuthor.keys():

    # for each word in our test sentence...

    for j  in preProcessedTestSentence:

        # find out how frequently the author used that word

        wordFreq = wordFreqByAuthor[i].freq(j)

        # and add a very small amount to every prob. so none of them are 0

        smoothedWordFreq = wordFreq + 0.000001

        # add the author, word and smoothed freq. to our dataframe

        output = pd.DataFrame([[i, j, smoothedWordFreq]], columns = ['author','word','probability'])

        testProbailities = testProbailities.append(output, ignore_index = True)



# empty dataframe for the probability that each author wrote the sentence

testProbailitiesByAuthor = pd.DataFrame(columns = ['author','jointProbability'])



# now let's group the dataframe with our frequency by author

for i in wordFreqByAuthor.keys():

    # get the joint probability that each author wrote each word

    oneAuthor = testProbailities.query('author == "' + i + '"')

    jointProbability = oneAuthor.product(numeric_only = True)[0]

    

    # and add that to our dataframe

    output = pd.DataFrame([[i, jointProbability]], columns = ['author','jointProbability'])

    testProbailitiesByAuthor = testProbailitiesByAuthor.append(output, ignore_index = True)



# and our winner is...

testProbailitiesByAuthor.loc[testProbailitiesByAuthor['jointProbability'].idxmax(),'author']