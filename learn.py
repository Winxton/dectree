import heapq
import numpy
import math
from joblib import Parallel, delayed
from scipy import sparse

global WORD_TO_ID_MAP
global DOC_ID_TO_LABEL_MAP
global DOC_WORD_PAIR_EXISTS

class TreeNode:
    def __init__(self, left, right, feature, pointEstimate):
        self.left = left
        self.right = right
        self.feature = feature
        self.pointEstimate = pointEstimate

class QueueNode:
    def __init__(self, node, informationGain, docList):
        self.treeNode = node
        self.informationGain = informationGain
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

        # majority Newsgroup 
        pointEstimate = 1 if alts > comps else 2
        return (feature, I - 0.5 * I_E1 - 0.5 * I_E2, pointEstimate)

    except IndexError: # happens if the word is not in either doc
        return (feature, 0, 0)

def getBestFeatureAndValue(wordList, docList):
    print "Getting best feature.."
    results = Parallel(n_jobs=4)(delayed(informationGainAverage)(word, docList) for word in wordList)
    bestWordInfoGainPair = max(results, key=lambda x: x[1])
    return bestWordInfoGainPair

""" Build the Decision Tree"""
def decisionTreeLearner(docList):
    feature, infoGain, estimate = getBestFeatureAndValue(docList)
    root = TreeNode(None, None, feature, estimate)

    # A priority queue of leaves by information gain
    pqLeaves = []

    # Initiate the priority queue with the root node
    queueNode = QueueNode(root, infoGain, docList)
    heapq.heappush(pqLeaves, queueNode)

    n = 0
    while (n < 10 and len(pqLeaves) > 0):
        bestQueueNode = heapq.heappop(pqLeaves)
        treeNode = bestQueueNode.treeNode

        # Split news by the feature 0 or 1
        L = [docId for docId in docList if DOC_WORD_PAIR_EXISTS[docId, WORD_TO_ID_MAP[treeNode.feature]] == 1]
        R = [docId for docId in docList if DOC_WORD_PAIR_EXISTS[docId, WORD_TO_ID_MAP[treeNode.feature]] == 0]

        n += 1


if __name__ == "__main__":

    # Read data

    A = numpy.loadtxt("trainData.txt")
    rows = A[:,0]
    cols = A[:,1]
    ones = numpy.ones(len(rows))
    DOC_WORD_PAIR_EXISTS = sparse.csr_matrix((ones, (rows, cols)))

    WORD_TO_ID_MAP = {}
    wordset = set()
    for idx, word in enumerate(open('words.txt', 'r')):
        filteredWord = word.strip('\n')
        WORD_TO_ID_MAP[filteredWord] = idx + 1
        wordset.update(filteredWord)

    docList = numpy.unique(rows)

    DOC_ID_TO_LABEL_MAP = {}
    for idx, label in enumerate(open('trainLabel.txt', 'r')):
        DOC_ID_TO_LABEL_MAP[idx + 1] = int(label)

    #print getBestFeatureAndValue(wordList, docList)