"""

Naive Bayes Classifer

Bayes classification is, of course, based on Bayes theorem:

        P(A) * P(B|A)
P(A|B) = ------------
            P(B)

It is called 'Naive' because we are making assumptions which may not be (probably aren't) true.
For example, mutual independence of all data points and follow a gaussian distribution

"""

import math

"""
Helpers for the normal distribution
"""

def mean(vector):
    return (sum(vector) / len(vector))
    
def variance(vector):
    avg = mean(vector)
    return sum((x - avg) ** 2 for x in vector) / len(vector)
    
def stddev(vector):
    return (variance(vector) ** .5)

"""
Definition of normal distribution
"""
def normal_probability(point, mean, stdev):
    	exp = math.exp(-((point-mean) ** 2)/(2*(stdev ** 2)))
    	return ((1 / (((2* math.pi * (stdev ** 2)) ** .5))) * exp)

"""
Next, we need a distribution for each parameter, for each class (in this case liver disease / no liver disease)
This is the 'model training' portion
"""

def make_distributions(class_0_vectors, class_1_vectors):
    class_0_distributions = []
    class_1_distributions = []
    
    _0 = []
    _1 = []

    #find std dev and mean for each parameter, for each class (stored as a tuple of (mean, stddev))
    for i in range(len(class_0_vectors[0])):
        for vector in class_0_vectors:
            _0.append(vector[i])

        for vector in class_1_vectors:
            _1.append(vector[i])
        
        class_0_distributions.append((mean(_0), stddev(_0)))
        class_1_distributions.append((mean(_1), stddev(_1)))
        _0 = []
        _1 = []
        
    return class_0_distributions, class_1_distributions
"""
If all the distributions are gaussian, this function can give us the probability for a single value
in our vector. To find the full value (because we assumed independence), we can simply multiply the
individual probabilites for each parameter. One note is that since probabilities are always
0 < P < 1, if you try to multiply too many together you could potentially underflow
(you can use a log to avoid this)
"""

def overall_probabilities(vector, zero_distributions, one_distributions):
    probability_0 = 1
    probability_1 = 1
    dist_0 = dist_1 = None
    for i in range(len(vector)):
        dist_0 = zero_distributions[i]
        dist_1 = one_distributions[i]
        probability_0 *= normal_probability(vector[i], dist_0[0], dist_0[1])
        probability_1 *= normal_probability(vector[i], dist_1[0], dist_1[1])
        
    return probability_0, probability_1