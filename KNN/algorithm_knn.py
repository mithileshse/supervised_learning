from numpy import *
import operator
#
def getData():
    features = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    classes = ['A', 'A', 'B', 'B']
    return features, classes

def test(inX, data, classes, k):
    no_of_rows = data.shape[0]
    diff_vector = tile(inX, (no_of_rows, 1)) - data
    sq_diff_vector = diff_vector ** 2
    sum_diff_vector = sq_diff_vector.sum(axis=1)
    distances = sum_diff_vector ** 0.5
    sorted_vector = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = classes[sorted_vector[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

group,labels=getData()
print(test([0, 0], group, labels, 3))
