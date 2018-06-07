__name__ = "Created by Ibrahim Gabr. April 2018."
from common_code import get_features, CLASSES
from hinge_loss import cost, display, extract_top_features
import numpy as np
from sklearn.metrics import accuracy_score
from copy import deepcopy
from pprint import pprint

def create_weight_dictionary_bigrams(lst):
    """
    Creates feature template of the form ((word1, word2), label): weight in a dictionary.
    
    Label is an integer value of 0,1 or 2.
    """
    feature_dict = {}
    for sentence in lst:
        words = sentence.split()
        label = int(words.pop()) # removes label from words in the sentence
        sent = [" ".join(words)]
        bigrams = [b for l in sent for b in zip(l.split(" ")[:-1], l.split(" ")[1:])]
        for pair in bigrams:
            feature_template = (pair, label)
            if pair not in feature_dict:
                feature_dict[pair] = {feature_template: 0}
            else:
                feature_dict[pair][feature_template] = 0
    return feature_dict

def extract_features_from_sentence_bigram(sentence):
    """
    This function extracts the bigrams from a sentence, the gold standard label and the 
    original sentence.
    """
    pairs = []
    words = sentence.split()
    gold_standard_label = int(words.pop())
    sent = [" ".join(words)]
    bigrams = [b for l in sent for b in zip(l.split(" ")[:-1], l.split(" ")[1:])]
    for pair in bigrams:
        pairs.append(pair)
    
    sentence = sent[0]
    return pairs, gold_standard_label, sentence

def feature_function_bigram(template, sentence, gold_standard_label):
    """
    Evaluates the feature function for a template on an input sentence and its associated gold standard label.
    """
    bigram = template[0]
    label = template[1]
    sentence_lst = [sentence]
    sentence_bigrams = [b for l in sentence_lst for b in zip(l.split(" ")[:-1], l.split(" ")[1:])]
    
    if (bigram in sentence_bigrams) and (label == gold_standard_label):
        return 1
    else:
        return 0

def costClassify_bigram(feature_dict, sent, gs_label):
    """
    Logically - it has to be this function. This is the only new thing
    You are adding to your code.
    
    re-write this code from scratch and look at the equation.
    """
    score_track = []
    for label in CLASSES:
        score = np.float_(0)
        cost_val = cost(gs_label, label)
        for feature_temp in feature_dict:
            feature_weight = feature_dict[feature_temp]
            feature_func_output = feature_function_bigram(feature_temp, sent, label)
            change_score = np.float_((feature_weight*feature_func_output))
            score += change_score
        score_track.append(score + cost_val)
    
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

def hinge_loss_bigram(weight_dict, sentence):
    words, gs_label, sent = extract_features_from_sentence_bigram(sentence)
    selected_features = get_features(words, weight_dict)
    classify_label = costClassify_bigram(selected_features, sent, gs_label)
    
    for feature_temp in selected_features:
        bigram = feature_temp[0]
        feature_weight = weight_dict[bigram][feature_temp]
        f_j_1 = feature_function_bigram(feature_temp, sent, gs_label)
        f_j_2 = feature_function_bigram(feature_temp, sent, classify_label)
        weight_update = feature_weight + 0.01*f_j_1 - 0.01*f_j_2
        weight_dict[bigram][feature_temp] = weight_update

def classify_bigram(feature_dict, sent):
    """
    Used in prediction function
    """
    score_track = []
    for label in CLASSES:
        score = np.float_(0)
        for feature_temp in feature_dict:
            feature_weight = feature_dict[feature_temp]
            feature_func_output = feature_function_bigram(feature_temp, sent, label)
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

def predict_bigram(test_data_lst, w_dict):
    """
    Given a sentence and the trained weights for our features - predict the sentiment of a sentence.
    
    Same for Hinge Loss and for Perceptron Loss ?
    """
    actual_labels = []
    predicted_labels = []
    
    for sent in test_data_lst:
        words, gs_label, sentence = extract_features_from_sentence_bigram(sent)
        selected_features = get_features(words, w_dict)
        classify_label = classify_bigram(selected_features, sentence)
        actual_labels.append(gs_label)
        predicted_labels.append(classify_label)
        
    return accuracy_score(actual_labels, predicted_labels)

def train_classifier_bigram(t_data, initial_weights, epochs, loss_func, dev_set, dev_test_set):
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
                acc = predict_bigram(dev_set, initial_weights) #testing every 20K sentences within an Epoch
                print(f"Accuracy after {sent_count} sentences in epoch {epoch} is : {acc}")
                if acc > best_acc: # only test if best accuracy we've seen
                    print(f"current best DEV accuracy is {acc}, previous best was {best_acc}")
                    best_acc = acc
                    dev_test_acc = predict_bigram(dev_test_set, initial_weights) # predict on DEV TEST
                    best_weights = deepcopy(initial_weights) # store copy of the best weights for performance on DEV TEST
                    print(f"Accuracy on Dev test is {dev_test_acc} in epoch {epoch} after {sent_count} sentences")
        acc_epoch = predict_bigram(dev_set, initial_weights) # check accuracy at the end of an epoch
        print(f"Accuracy at the end of epoch {epoch} is {acc_epoch}")
        if acc_epoch > best_acc:
            print(f"current best DEV accuracy is {acc_epoch}, previous best was {best_acc}")
            best_acc = acc_epoch
            dev_test_acc = predict_bigram(dev_test_set, initial_weights)
            print(f"Accuracy on DEV TEST is {dev_test_acc} in epoch {epoch} after {sent_count} sentences")
        print()
    stats = [best_acc, dev_test_acc]
    
    return stats, best_weights

def predict_error_analysis_bigram(test_data_lst, weight_dict):
    results_lst = []
    
    for sent in test_data_lst:
        words, gs_label, sentence = extract_features_from_sentence_bigram(sent)
        selected_features = get_features(words, weight_dict)
        classify_label = classify_bigram(selected_features, sentence)
        if classify_label != gs_label:
            results_lst.append((sentence, classify_label, gs_label))
    return results_lst