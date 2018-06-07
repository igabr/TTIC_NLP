__name__ = "Created by Ibrahim Gabr. April 2018."
from common_code import extract_features_from_sentence, get_features, classify, feature_function

def perceptron_loss(weight_dict, sentence):
    """
    This code represents the perceptron loss function as specified in the assignment.

    Note how the function does not return anything. This is because out weights are stored in a dictionary.

    Manipulations to the weight dictionary below will occur in-place as the dictionary has a global shared memory location.
    """
    words, gs_label, sent = extract_features_from_sentence(sentence)
    selected_features = get_features(words, weight_dict)
    classify_label = classify(selected_features, sent) # classify label only calculated once at the beginning of a training example.
    
    for feature_temp in selected_features:
        word = feature_temp[0]
        feature_weight = weight_dict[word][feature_temp]
        f_j_1 = feature_function(feature_temp, sent, gs_label)
        f_j_2 = feature_function(feature_temp, sent, classify_label)
        weight_update = feature_weight + 0.01*f_j_1 - 0.01*f_j_2 #sub-gradient descent step. Step size is fixed to 0.01
        weight_dict[word][feature_temp] = weight_update