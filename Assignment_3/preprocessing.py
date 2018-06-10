__name__ = "Created by Ibrahim Gabr. May 2018."

from utils import *

def pre_process_file(training_lst, file_lst, embedding_lst):
    """
    Inputs:
        training_lst : training file used to create POS dictionary.
        file_lst: The file that needs to be pre-processed
        embedding_lst: The pre-trained embedding file.
    """
    POS_DICT = gen_POS_dict(training_lst)
    embedding_dictionary = generate_embedding_dictionary(embedding_lst)
    tweets, targets = get_tweets_and_targets(file_lst)
    
    all_training_trigrams = list(map(generate_trigrams, tweets))
    
    
    embeddings_lst = []
    one_hot_lst = []

    for outer_index, sublist in enumerate(all_training_trigrams):
        target_lst = targets[outer_index]
        for index, trigram in enumerate(sublist):
            target_value = target_lst[index] # the target associated with this trigram
            start, center, end = trigram # the actual trigram

            target_vec = generate_one_hot_vector(target_value, POS_DICT) # one hot representation of true target label
            embedding_vec = get_embedding_vector(start, center, end, embedding_dictionary) # concatenated embedding

            one_hot_lst.append(target_vec)
            embeddings_lst.append(embedding_vec)
    
    input_matrix = np.stack(embeddings_lst)
    
    target_matrix = np.stack(one_hot_lst)
    
    return input_matrix, target_matrix
