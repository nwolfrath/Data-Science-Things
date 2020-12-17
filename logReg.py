"""
This is a logistic regression built by hand (not even using numPy!).
I tried to give a brief overview of the math as I went.
"""
import math
import random

def dotProduct(vector1, vector2):
    sum = 0
    pairs = zip(vector1, vector2)
    for term1, term2 in pairs:
        sum += (term1 * term2)
    return sum

def addVectors(vector1, vector2):
    ret = []
    pairs = zip(vector1, vector2)
    for term1, term2 in pairs:
        ret.append(term1 + term2)

    return ret

"""This is just the definition of the logistic function"""

def logistic(n):
    return 1 / (1 + math.exp(-n))

"""And the first derivative of the logistic function"""

def logisticDerivative(n):
    return logistic(n) * (1-logistic(n))

"""
We can make a model with this according to y = logistic(x * W) + error where W is some weight
Where Y is 1 with probability logistic(x * W) (call this p)
and Y is 0 with probability 1 - logistic(x * W) (call this q)

The conditional probability density function of Y can be written as follows:

P(Y = y | X = x, W) = logistic(X * W)^y * (1 - logistic (X * W) ^(1-Y))
This can be confirmed as P(Y = 1) = p and P(Y = 0) = q



After a significant amount of pain and browsing wikipedia, I discovered it is more common to compute
the 'log likelihood' (makes taking partial derivatives much much easier and prevents possible
underflow by using summations rather than products). Since log is strictly increasing,
this won't affect when we maximize the likelihood function.

This can be written:

Y * log(logistic(X* W)) + (1 - Y) * log(1 - logistic (X * W))
"""

def logLikelihood(x, y, weights):
    weightedX = dotProduct(x, weights)
    return ((y * math.log(logistic(weightedX))) + ((1-y)* math.log(1 - logistic(weightedX))))


"""
Since we are dealing with probabilities, (if we make an assumption that all data points are
independant), The overall likelihood is just the product of all the individual probabilities.
since log(x * y) = log(x) + log(y), when using the log likelihood, we can compute this
as a sum rather than as a product.
"""

def overallLikelihood(x, y, weights):
    sum = 0
    combined = zip(x,y)
    for xi, yi in combined:
        sum += logLikelihood(xi, yi, weights)

    return sum


"""
In order to maximize this, we will need to use some calculus. We need to compute the partial
derivative of the likelihood with respect to each weight. The gradient function points in the direction
of max increase for a multi-dimensional surface. We will be following this function
to find the (local) maximum likelihood given some vector of inputs. This is called gradient descent,
although we will be going uphill rather than down.

Got some math knowledge from here:
https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/pdfs/40%20LogisticRegression.pdf
"""

def singlePointPartial(x, y, weights, i):

    #first get the partial for each point in an individual term
    return (y - logistic(dotProduct(x, weights))) * x[i]

def singlePointGradient(x, y, weights):
    ret = []
    for i, _ in enumerate(weights):
        ret.append(singlePointPartial(x, y, weights, i))

    return ret
    
def gradient(x, y, weights):
    #Derivitive of the sum is just the sum of derivatives
    sum = singlePointGradient(x[0], y[0], weights)
    
    for xi, yi in zip(x[1:],y[1:]):
        sum = addVectors(sum, singlePointGradient(xi, yi, weights))

    return sum


"""
We now have our gradient function, we are going to use stochastic gradient descent
I didn't generalize this, but you could pass in a target and gradient function
to use this in more cases
"""
    
def stochasticGradientAscent(x, y, initialWeights, initialStepSize, iterations):
    newWeights = initialWeights
    stepSize = initialStepSize
    bestScore = float("-inf")
    bestWeights = None
    shuffledData = list(zip(x,y))
    

    for _ in range(iterations):
        score = overallLikelihood(x, y, newWeights)
        
        if (score > bestScore):
            bestScore = score
            bestWeights = newWeights
            #This is because we change step size if we can't find improvement, reset here.
            stepSize = initialStepSize

        else:
            #If we can't improve at our current step size, try a smaller one
            stepSize *= .9

        #for fairly small data sets, we can take a step for each data point
        #for bigger ones, take some random subset
        for a, b in shuffledData:
            gradient = singlePointGradient(a, b, newWeights)
            newWeights = addVectors(newWeights, [stepSize * term for term in gradient])

    return bestWeights

    
    #RESCALE DATA!!!


#Some stuff just for the presentation
def drawLogistic():
    import matplotlib.pyplot as plt
    x = [i for i in range(-50,50)]
    y = [logistic(xi) for xi in x]
    plt.plot(x,y)
    plt.show()
