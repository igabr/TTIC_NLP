__name__ = "Created by Ibrahim Gabr. April 2018."
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from pprint import pprint
from copy import deepcopy

CLASSES = [0,1,2] #these are the possible classes for predictions.

"""
This script contains code that is used for BOTH the perceptron and hinge loss sections of the assignment.

Code that applied only for the perceptron/hinge section of the assignment is located in perceptron_loss.py/hinge_loss.py
"""

def create_weight_dictionary(lst):
    """
    Creates feature template of the form (word, label): weight in a dictionary.
    
    Label is an integer value of 0,1 or 2.

    This function is used for both Perceptron Loss and Hinge Loss sections.
    """
    feature_dict = {}
    for sentence in lst:
        words = sentence.split()
        label = int(words.pop()) # removes label from words in the sentence
        for word in words:
            feature_template = (word, label) #we are only creating feature templates that are possible from the training data. Not all possible combos.
            if word not in feature_dict:
                feature_dict[word] = {feature_template:0}
            else:
                feature_dict[word][feature_template] = 0
    return feature_dict

def extract_features_from_sentence(sentence):
    """
    This function extracts the unigrams from a sentence, the gold standard label and the 
    original sentence.
    """
    word_lst = []
    words = sentence.split()
    gold_standard_label = int(words.pop())
    
    for word in words:
        word_lst.append(word)
    
    sentence = " ".join(words)
    
    return word_lst, gold_standard_label, sentence

def feature_function(template, sentence, gold_standard_label):
    """
    Evaluates the feature function for a template on an input sentence and its associated gold standard label.

    Used in Perceptron Loss and Hinge Loss
    """
    if (template[0] in sentence) and (template[1] == gold_standard_label):
        return 1
    else:
        return 0

def get_features(word_lst, weight_dict):
    """
    This function will ensure we only loop over labels that are present in our sentence.

    This way, we are not looping over the entire dictionary - we are only concerned with features that will "fire".
    
    This function is used in both Perceptron Loss and Hinge Loss.
    """
    selected_feature_lst = []
    for word in word_lst:
        if word in weight_dict:
            selected_templates = weight_dict[word] # get the dictionary of templates for a word
            selected_feature_lst.append(selected_templates)
        else:
            continue #if we haven't seen a word before (in training data) ignore it - same as weight being set to 0.
    return { k: v for d in selected_feature_lst for k, v in d.items() } #combine all templates and their weights into a dictionary.

def classify(feature_dict, sent):
    """
    This function represents the classifier as stated in the homework.

    This function is used in both Perceptron Loss and Hinge Loss.
    """
    score_track = []
    for label in CLASSES:
        score = np.float_(0)
        for feature_temp in feature_dict:
            feature_weight = feature_dict[feature_temp]
            feature_func_output = feature_function(feature_temp, sent, label)
            change_score = np.float_(feature_weight* feature_func_output)
            score += change_score
        score_track.append(score)
    
    score_track = np.asarray(score_track)
    y0,y1,y2 = score_track
    
    if y0 == y1 == y2:
        return 0
    elif y0 == y1 or y0 == y2:
        return 0
    elif y1 == y2:
        return 1
    else:
        return np.argmax(score_track)

def predict(test_data_lst, w_dict):
    """
    Given a list of sentences (from test data) and the current weight dictionary 
    predict the sentiment of every sentence in the list.

    This function returns an accuracy score between actual labels and predicted labels.
    
    This function is used in both Perceptron Loss and Hinge Loss.

    This function trains a classifier given a list of sentences as training data, the initialized weight dictionary and a specified loss function.

    This function will return a list called stats which contains two variables:
        - best_acc: best accuracy value seen on DEV while training
        - dev_test_acc = best accuracy seen on DEV TEST while training
    
    This function also returns a dictionary called best_weights:
        - best_weights: There are the weights that lead to the highest accuracy on DEV TEST while training.
    
    This function is used for both Perceptron and Hinge Loss.
    """
    actual_labels = []
    predicted_labels = []
    
    for sent in test_data_lst:
        words, gs_label, sentence = extract_features_from_sentence(sent)
        selected_features = get_features(words, w_dict)
        classify_label = classify(selected_features, sentence)
        actual_labels.append(gs_label)
        predicted_labels.append(classify_label)
        
    return accuracy_score(actual_labels, predicted_labels)

def train_classifier(t_data, initial_weights, epochs, loss_func, dev_set, dev_test_set):
    """
    This function trains a classifier given a list of sentences as training data, the initialized weight dictionary and a specified loss function.

    Inputs:
        - t_data: Training Data - list of sentences.
        - initial_weights - initialized weight dictionary.
        - epochs: numbers of epochs to iterate through
        - loss_func: loss_function to use.
        - dev_set: The in-sample test set
        - dev_test_set: The out-sample/hold out test set.

    This function will return a list called stats which contains two variables:
        - best_acc: best accuracy value seen on DEV while training
        - dev_test_acc = best accuracy seen on DEV TEST while training
    
    This function also returns a dictionary called best_weights:
        - best_weights: There are the weights that lead to the highest accuracy on DEV TEST while training.
    
    This function is used for both Perceptron and Hinge Loss.
    """
    best_weights = None
    best_acc = 0
    for epoch in range(epochs):
        print(f"---starting epoch {epoch}---")
        sent_count = 0
        for sent in t_data:
            loss_func(initial_weights, sent)
            sent_count +=1
            if sent_count % 20000 == 0:
                acc = predict(dev_set, initial_weights) #testing every 20K sentences within an Epoch
                print(f"Accuracy after {sent_count} sentences in epoch {epoch} is : {acc}")
                if acc > best_acc: # only test if best accuracy we've seen
                    print(f"current best DEV accuracy is {acc}, previous best was {best_acc}")
                    best_acc = acc
                    dev_test_acc = predict(dev_test_set, initial_weights) # predict on DEV TEST
                    best_weights = deepcopy(initial_weights) # store copy of the best weights for performance on DEV TEST
                    print(f"Accuracy on Dev test is {dev_test_acc} in epoch {epoch} after {sent_count} sentences")
        acc_epoch = predict(dev_set, initial_weights) # check accuracy at the end of an epoch
        print(f"Accuracy at the end of epoch {epoch} is {acc_epoch}")
        if acc_epoch > best_acc:
            print(f"current best DEV accuracy is {acc_epoch}, previous best was {best_acc}")
            best_acc = acc_epoch
            dev_test_acc = predict(dev_test_set, initial_weights)
            print(f"Accuracy on DEV TEST is {dev_test_acc} in epoch {epoch} after {sent_count} sentences")
        print()
    stats = [best_acc, dev_test_acc]
    
    return stats, best_weights