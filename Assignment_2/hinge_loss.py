__name__ = "Created by Ibrahim Gabr. April 2018."
import numpy as np
from common_code import feature_function, extract_features_from_sentence, get_features, CLASSES, classify

def cost(gs, lbl):
    """
    This function represents the cost component of hinge loss.

    If the gold standard label and provided label do not match, 1 is returned.

    Mathematically, this is an indicator function.
    """
    if gs != lbl:
        return 1
    else:
        return 0

def costClassify(feature_dict, sent, gs_label):
    """
    Logically - it has to be this function. This is the only new thing
    You are adding to your code.
    
    re-write this code from scratch and look at the equation.

    This is similar to the classify function used in perceptron loss, only we have added the cost component.
    """
    score_track = []
    for label in CLASSES:
        score = np.float_(0)
        cost_val = cost(gs_label, label)
        for feature_temp in feature_dict:
            feature_weight = feature_dict[feature_temp]
            feature_func_output = feature_function(feature_temp, sent, label)
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

def hinge_loss(weight_dict, sentence):
    """
    This code represents the hinge loss function as specified in the assignment.

    Note how the function does not return anything. This is because out weights are stored in a dictionary.

    Manipulations to the weight dictionary below will occur in-place as the dictionary has a global shared memory location.
    """
    words, gs_label, sent = extract_features_from_sentence(sentence)
    selected_features = get_features(words, weight_dict)
    classify_label = costClassify(selected_features, sent, gs_label) # classify label only calculated once at the beginning of a training example. Note use of costClassify and not classify
    
    for feature_temp in selected_features:
        word = feature_temp[0]
        feature_weight = weight_dict[word][feature_temp]
        f_j_1 = feature_function(feature_temp, sent, gs_label)
        f_j_2 = feature_function(feature_temp, sent, classify_label)
        weight_update = feature_weight + 0.01*f_j_1 - 0.01*f_j_2 #sub-gradient descent step. Step size is fixed to 0.01
        weight_dict[word][feature_temp] = weight_update

# Section 1.3 code below

def display(x):
    return print(x, "\n")

def extract_top_features(weight_dict):
    for label in CLASSES:
        all_weights = []
        for word in weight_dict:
            template = (word, label) # creating features
            try: #possible that some feature, label combinations were not seen in training data.
                weight = weight_dict[word][template]
                all_weights.append((template,weight))
            except KeyError:
                continue
        top_features = sorted(all_weights, key=lambda x: x[1], reverse=True)[:10]
        print(f"The top features for label {label} are: \n")
        list(map(display, top_features))
        print()

# Section 1.4 code below

def predict_error_analysis(test_data_lst, weight_dict):
    results_lst = []
    
    for sent in test_data_lst:
        words, gs_label, sentence = extract_features_from_sentence(sent)
        selected_features = get_features(words, weight_dict)
        classify_label = classify(selected_features, sentence)
        if classify_label != gs_label:
            results_lst.append((sentence, classify_label, gs_label))
    return results_lst