import heapq
import numpy
import math
from joblib import Parallel, delayed
from scipy import sparse
import matplotlib.pyplot as plt

global WORD_TO_ID_MAP
global DOC_ID_TO_LABEL_MAP
global DOC_WORD_PAIR_EXISTS
global CACHE

class TreeNode:
    def __init__(self, left, right, feature, infoGain, pointEstimate):
        self.left = left
        self.right = right
        self.feature = feature
        self.infoGain = infoGain
        self.pointEstimate = pointEstimate

    def classification(self):
        return 'alt.atheism' if self.pointEstimate == 1 else 'comp.graphics'

    def __str__(self, level=0):
        ret = "\t"*level + repr(self)+"\n"
        if self.left is not None:
            ret += self.left.__str__(level+1)
        if self.right is not None:
            ret += self.right.__str__(level+1)
        return ret

    def __repr__(self):
        return '<%s, %s, %s>' % \
        (self.classification(), self.feature, str(self.infoGain))

class QueueNode:
    def __init__(self, node, informationGain, wordset, docList):
        self.treeNode = node
        self.informationGain = informationGain
        self.wordset = wordset
        self.docList = docList

    def __cmp__(self, other):
        return cmp(self.informationGain, other.informationGain)

def getInfo(n, m):
    if (n + m == 0):
        return 1
    p1 = 1.0 * n / (n+m)
    p2 = 1 - p1
    p1 = 0 if p1 == 0 else p1*math.log(p1, 2)
    p2 = 0 if p1 == 0 else p2*math.log(p2, 2)
    return -p1-p2

def informationGainAverage(feature, docList):
    countFeatureT_AND_Alt = 0
    countFeatureF_AND_Alt = 0
    countFeatureT_AND_Comp = 0
    countFeatureF_AND_Comp = 0
    try:
        for docId in docList:
            if (DOC_WORD_PAIR_EXISTS[docId, WORD_TO_ID_MAP[feature]] == 1):
                if (DOC_ID_TO_LABEL_MAP[docId] == 1):
                    countFeatureT_AND_Alt += 1
                else:
                    countFeatureT_AND_Comp += 1
            else:
                if (DOC_ID_TO_LABEL_MAP[docId] == 1):
                    countFeatureF_AND_Alt += 1
                else:
                    countFeatureF_AND_Comp += 1

        alts = countFeatureT_AND_Alt + countFeatureF_AND_Alt
        comps = countFeatureT_AND_Comp + countFeatureF_AND_Comp
        I = getInfo(alts, comps)
        I_E1 = getInfo(countFeatureT_AND_Alt, countFeatureT_AND_Comp)
        I_E2 = getInfo(countFeatureF_AND_Alt, countFeatureF_AND_Comp)

        N_1 = countFeatureT_AND_Alt + countFeatureT_AND_Comp
        N_2 = countFeatureF_AND_Alt + countFeatureF_AND_Comp
        N = N_1 + N_2

        # majority Newsgroup 
        pointEstimate = 1 if alts > comps else 2
        return (feature, I - (1.0*N_1/N)*I_E1 - (1.0*N_2/N)*I_E2, pointEstimate)

    except IndexError: # happens if the word is not in either doc
        return (feature, 0, 0)

def getBestFeatureAndValue(wordset, docList):
    #print "Getting best feature.."

    if (len(CACHE)>0):
        bestWordInfoGainEstimateTuple = CACHE.pop(0)
    else:
        results = Parallel(n_jobs=4)(delayed(informationGainAverage)(word, docList) for word in wordset)
        bestWordInfoGainEstimateTuple = max(results, key=lambda x: x[1])

    print bestWordInfoGainEstimateTuple
    return bestWordInfoGainEstimateTuple

""" Build the Decision Tree"""
def decisionTreeLearner(wordset, docList, maxNodes):
    print "Learning..."
    feature, infoGain, estimate = getBestFeatureAndValue(wordset, docList)
    root = TreeNode(None, None, feature, infoGain, estimate)

    # A priority queue of leaves by information gain
    pqLeaves = []

    # Initiate the priority queue with the root node
    queueNode = QueueNode(root, infoGain, wordset.difference(set([feature])), docList)
    heapq.heappush(pqLeaves, queueNode)

    testDocList, DocWordMatrix, DocLabelTestData = getDataInfo('testData.txt', 'testLabel.txt')
    percentagesCorrect = []

    n = 0
    while (n < maxNodes and len(pqLeaves) > 0):
        bestQueueNode = heapq.heappop(pqLeaves)
        treeNode = bestQueueNode.treeNode

        # Split news by the feature 0 or 1
        # Stop splitting if all documents are in the same group
        L = [docId for docId in bestQueueNode.docList if DOC_WORD_PAIR_EXISTS[docId, WORD_TO_ID_MAP[treeNode.feature]] == 1]
        if len(set([DOC_ID_TO_LABEL_MAP[docId] for docId in L])) == 1: # All same group
            leftTreeNode = TreeNode(None, None, None, 0, DOC_ID_TO_LABEL_MAP[L[0]])
        else: 
            feature, infoGain, estimate = getBestFeatureAndValue(bestQueueNode.wordset, L)
            leftTreeNode = TreeNode(None, None, feature, infoGain, estimate)
            leftQueueNode = QueueNode(leftTreeNode, infoGain, bestQueueNode.wordset.difference(set([feature])), L)
            heapq.heappush(pqLeaves, leftQueueNode)
        treeNode.left = leftTreeNode

        R = [docId for docId in bestQueueNode.docList if DOC_WORD_PAIR_EXISTS[docId, WORD_TO_ID_MAP[treeNode.feature]] == 0]
        if len(set([DOC_ID_TO_LABEL_MAP[docId] for docId in R])) == 1: # All same group
            rightTreeNode = TreeNode(None, None, None, 0, DOC_ID_TO_LABEL_MAP[R[0]])
        else: 
            feature, infoGain, estimate = getBestFeatureAndValue(bestQueueNode.wordset, R)
            rightTreeNode = TreeNode(None, None, feature, infoGain, estimate)
            rightQueueNode = QueueNode(rightTreeNode, infoGain, bestQueueNode.wordset.difference(set([feature])), R) 
            heapq.heappush(pqLeaves, rightQueueNode)
        treeNode.right = rightTreeNode

        #print "%s - %i Split %i %i" % (treeNode.feature, len(bestQueueNode.docList), len(L), len(R))

        # Classify Here to generate chart
        
        numCorrect = sum( classify(docId, root, DocWordMatrix) == DocLabelTestData[docId] for docId in testDocList)
        percentage = 1.0*numCorrect/len(testDocList)
        percentagesCorrect.append( percentage )
        print n, percentage
        
        n += 1

    
    plt.title('Training data classifications')
    plt.xlabel('Number of Nodes Used')
    plt.ylabel('Percentage Correct')
    xAxisNodes = [i+1 for i in range(0,len(percentagesCorrect))]
    plt.plot(xAxisNodes, percentagesCorrect, 'ro')
    plt.show()
    

    return root

def classify(docId, DecisionTree, DocWordMatrix):
    cur = DecisionTree

    while (cur!=None and cur.feature != None):
        classification = cur.pointEstimate

        # go left if it contains the word, otherwise go right
        if (DocWordMatrix[docId, WORD_TO_ID_MAP[cur.feature]] == 1):
            cur = cur.left
        else:
            cur = cur.right

    return classification

def getDataInfo(dataFile, labelFile):
    # Read data
    A = numpy.loadtxt(dataFile)
    rows = A[:,0]
    cols = A[:,1]
    ones = numpy.ones(len(rows))
    DocWordMatrix = sparse.csr_matrix((ones, (rows, cols)))
    docList = numpy.unique(rows)

    DocLabelTestData = {}
    for idx, label in enumerate(open(labelFile, 'r')):
        DocLabelTestData[idx + 1] = int(label)

    return docList, DocWordMatrix, DocLabelTestData

def generateTestDataGraph(DecisionTree):
    print "Classifying Test Data..."

    docList, DocWordMatrix, DocLabelTestData = getDataInfo('testData.txt', 'testLabel.txt')
    percentagesCorrect = []
    
    for nodes in range(0,100):
        numCorrect = 0
        for docId in docList:
            classification = classify(docId, nodes+1, DecisionTree, DocWordMatrix)
            if (classification == DocLabelTestData[docId]):
                numCorrect += 1
        percentage = 1.0*numCorrect/len(docList)
        percentagesCorrect.append( percentage )
        print nodes + 1, percentage

    plt.title('Test data classifications')
    plt.xlabel('Number of Nodes Used')
    plt.ylabel('Percentage Correct')
    xAxisNodes = [i+1 for i in range(0,100)]
    plt.plot(xAxisNodes, percentagesCorrect, 'ro')
    plt.show()

if __name__ == "__main__":
    import cache
    import cacheWeighted

    CACHE = cacheWeighted.CACHE

    # Read data
    WORD_TO_ID_MAP = {}
    wordset = set()
    for idx, word in enumerate(open('words.txt', 'r')):
        filteredWord = word.strip('\n')
        WORD_TO_ID_MAP[filteredWord] = idx + 1
        wordset.add(filteredWord)

    docList, DOC_WORD_PAIR_EXISTS, DOC_ID_TO_LABEL_MAP = getDataInfo('trainData.txt', 'trainLabel.txt')

    root = decisionTreeLearner(wordset, docList, 100)
    #generateTrainingDataGraph(docList, root)
    #generateTestDataGraph(root)