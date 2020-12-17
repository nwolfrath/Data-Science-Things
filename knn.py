"""

K Nearest Neighbors Classifier

"""

import math

"""
In order to classify, we first need some type of notion of how far away two data points from each other. In other words, the Euclidean distance.
"""

def euclidean_distance(vector1: list, vector2: list):
    if(len(vector1) != len(vector2)):
        return None
    total = 0
    #subtract one to ignore the label
    for i in range(len(vector1)):
        total+=((vector1[i]- vector2[i]) ** 2)
        
    return total ** .5
        

"""
The k nearest neighbors method is simple. From the training data, find the geometrically closest k points, and assign the type that most of those have.
"""
def nearest_neighbors(k, dataset: list, vector: list):
    neighbors = []
    for vector_i in dataset:
        distance = euclidean_distance(vector_i[0:-1], vector)
        neighbor = (vector_i[-1], distance)
        neighbors.append(neighbor)
    
    sorted_neighbors = sorted(neighbors, key=lambda neighbor_i: neighbor_i[1])
    return majority_rules([i for i, j in sorted_neighbors[0:k]])


def majority_rules(votes):
    """
    this just returns the type with more 'votes'. currently written for a binary classification, but could easily be extended to a more generalized form.
    """
    option1 = votes[0]
    option2 = None
    count = 0
    #find out what the other candidate is
    for vote in votes:
        if(option2 is None and vote != option1):
            option2 = vote
            break
    
    for vote in votes:
        if vote == option1:
            count += 1
            
    if count > (len(votes) / 2):
        return option1
    
    else:
        return option2
        
