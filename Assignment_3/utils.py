__name__ = "Created by Ibrahim Gabr. May 2018."

import numpy as np

def gen_POS_dict(tweet_lst):
    """
    Generates the POS tag dictionary from training data.

    This is the dictionary that will all for the creation of a one hot vector of (1,25).

    This dictionary will be references with using DEVTEST and DEV in order to maintain indicies.
    """
    POS_DICT = {}
    TAG_LST = []

    for index, word in enumerate(tweet_lst):
        if word == "\n":#ignore newlines.
            continue
        tag = word.split("\t")[-1].strip()
        if tag not in TAG_LST:# using a list to ensure reproducible indicies.
            TAG_LST.append(tag)
    
    for index, value in enumerate(TAG_LST):
        if value not in POS_DICT:
            POS_DICT[value] = index
    
    return POS_DICT


def generate_one_hot_vector(tag, d):
    """
    Returns the one hot vector for a given tag.
    """
    ind = d[tag]
    vector = np.zeros((1,25))
    vector[0][ind] = 1
    return vector

def get_tweets_and_targets(tweet_lst):
    """
    Returns:
        tweets - list of lists where each sub list represent a single tweet with only the text of the tweet.
        targets - list of lists where each sub list represents the associated tag for each word in a tweet.
        
        each entry in tweets and targets have the same length as there is a one-to-one relation.
    """
    start_index = 0

    tweets = []
    targets = []

    for index, value in enumerate(tweet_lst):
        word_lst = []
        target_lst = []
        if value == "\n":
            tweet = tweet_lst[start_index:index+1]
            start_index = index + 1
            tweet.pop() # remove \n from the end of a tweet.
            for word in tweet:
                if "\xa0" in word:
                    w = " "
                    tag = word.split()[-1]
                    word_lst.append(w)
                    target_lst.append(tag)
                else:
                    w, tag = word.split()
                    word_lst.append(w)
                    target_lst.append(tag)
            tweets.append(word_lst)
            targets.append(target_lst)
    return tweets, targets

def generate_trigrams(tweet_lst):
    """
    Given a tweet in list for, generates all valid trigrams.
    """
    assert type(tweet_lst) == list, "Your tweet must be a list of words."
    
    trigram_lst = []
    
    if len(tweet_lst) == 1: # rare case.
        start_word = "<s>"
        center_word = tweet_lst[0]
        subseq_word = "</s>"
        trigram_lst.append((start_word, center_word, subseq_word))
    else:
        for index, value in enumerate(tweet_lst):
            if index == 0:
                start_word = "<s>"
                center_word = tweet_lst[index]
                subseq_word = tweet_lst[index+1]
            elif index == len(tweet_lst) - 1:
                start_word = tweet_lst[index-1]
                center_word = tweet_lst[index]
                subseq_word = "</s>"
            else:
                start_word = tweet_lst[index-1]
                center_word = tweet_lst[index]
                subseq_word = tweet_lst[index+1]

            trigram_lst.append((start_word, center_word, subseq_word))
    
    return trigram_lst

def generate_embedding_dictionary(embed_lst):
    """
    Generates embeddings for data in the training set.
    """
    embedding_dictionary = {}
    
    for index, embed in enumerate(embed_lst):
        seperation = embed.split("\t")
        word = seperation[0]
        weights = seperation[-1]
        weights = np.array(list(map(float,weights.strip().split()))).reshape(1,50)
        if word not in embedding_dictionary:
            embedding_dictionary[word] = weights
    
    return embedding_dictionary

def get_embedding_vector(start, center, end, d):
    """
    Returns a (1,150) vector representing a trigram embedding.
    """
    
    start_w = d.get(start, d["UUUNKKK"])
    
    center_w = d.get(center, d["UUUNKKK"])
    
    end_w = d.get(end, d["UUUNKKK"])
    
    return np.c_[start_w, center_w, end_w]

        