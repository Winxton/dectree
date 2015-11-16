import heapq
import numpy
import math
from joblib import Parallel, delayed
from scipy import sparse

global WORD_TO_ID_MAP
global DOC_ID_TO_LABEL_MAP
global DOC_WORD_PAIR_EXISTS
global CACHE

class TreeNode:
    def __init__(self, left, right, feature, pointEstimate):
        self.left = left
        self.right = right
        self.feature = feature
        self.pointEstimate = pointEstimate

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

        # majority Newsgroup 
        pointEstimate = 1 if alts > comps else 2
        return (feature, I - 0.5 * I_E1 - 0.5 * I_E2, pointEstimate)

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
def decisionTreeLearner(wordset, docList):
    feature, infoGain, estimate = getBestFeatureAndValue(wordset, docList)
    root = TreeNode(None, None, feature, estimate)

    # A priority queue of leaves by information gain
    pqLeaves = []

    # Initiate the priority queue with the root node
    queueNode = QueueNode(root, infoGain, wordset.difference(set([feature])), docList)
    heapq.heappush(pqLeaves, queueNode)

    n = 0
    while (n < 100 and len(pqLeaves) > 0):
        bestQueueNode = heapq.heappop(pqLeaves)
        treeNode = bestQueueNode.treeNode

        # Split news by the feature 0 or 1
        # Stop splitting if all documents are in the same group
        L = [docId for docId in bestQueueNode.docList if DOC_WORD_PAIR_EXISTS[docId, WORD_TO_ID_MAP[treeNode.feature]] == 1]
        if len(set([DOC_ID_TO_LABEL_MAP[docId] for docId in L])) == 1: # All same group
            leftTreeNode = TreeNode(None, None, None, DOC_ID_TO_LABEL_MAP[L[0]])
        else: 
            feature, infoGain, estimate = getBestFeatureAndValue(bestQueueNode.wordset, L)
            leftTreeNode = TreeNode(None, None, feature, estimate)
            leftQueueNode = QueueNode(leftTreeNode, infoGain, bestQueueNode.wordset.difference(set([feature])), L)
            heapq.heappush(pqLeaves, leftQueueNode)
        treeNode.left = leftTreeNode

        R = [docId for docId in bestQueueNode.docList if DOC_WORD_PAIR_EXISTS[docId, WORD_TO_ID_MAP[treeNode.feature]] == 0]
        if len(set([DOC_ID_TO_LABEL_MAP[docId] for docId in R])) == 1: # All same group
            rightTreeNode = TreeNode(None, None, None, DOC_ID_TO_LABEL_MAP[R[0]])
        else: 
            feature, infoGain, estimate = getBestFeatureAndValue(bestQueueNode.wordset, R)
            rightTreeNode = TreeNode(None, None, feature, estimate)
            rightQueueNode = QueueNode(rightTreeNode, infoGain, bestQueueNode.wordset.difference(set([feature])), R) 
            heapq.heappush(pqLeaves, rightQueueNode)
        treeNode.right = rightTreeNode

        print "%s - %i Split %i %i" % (treeNode.feature, len(bestQueueNode.docList), len(L), len(R))
        n += 1

    return root

if __name__ == "__main__":

    CACHE = [
        ('atheism', 0.5007250672835606, 2), 
        ('evidence', 0.4976069389728536, 2),
        ('moral', 0.4935248144434943, 2),
        ('islam', 0.4883180190497184, 2),
        ('bible', 0.4832592298621504, 2),
        ('solntze', 0.4778028659949304, 2),
        ('benedikt', 0.4703020274726135, 2),
        ('tek', 0.4635862253959928, 2),
        ('mathew', 0.4571242050545491, 2),
        ('religion', 0.45081764719703954, 2),
        ('umd', 0.44256338371231796, 2),
        ('christian', 0.43504841945251177, 2),
        ('bobby', 0.42735466238641695, 2),
        ('morality', 0.41957918996161314, 2),
        ('punishment', 0.4118292813230854, 2),
        ('sandvik', 0.40510372187834137, 2),
        ('isc', 0.3969785771726777, 2),
        ('rice', 0.39012639554238676, 2),
        ('buphy', 0.38285597441536123, 2),
        ('cco', 0.37401257433418356, 2),
        ('tammy', 0.36562364247940177, 2),
        ('kill', 0.35907189841799836, 2),
        ('cwru', 0.35092190212233887, 2),
        ('atheists', 0.34489810889817, 2),
        ('jesus', 0.3386307157415276, 2),
        ('evolution', 0.33066424965016056, 2),
        ('tao', 0.3252025149319054, 2),
        ('darice', 0.3180278988134896, 2),
        ('liar', 0.3136303173811449, 2),
        ('religious', 0.3091204096235898, 2),
        ('mantis', 0.3044942500650024, 2),
        ('alexia', 0.2997476973539736, 2),
        ('stronger', 0.29487637617232865, 2),
        ('civil', 0.2898756570095833, 2),
        ('objective', 0.28474063347307255, 2),
        ('kmr', 0.2794660967401992, 2),
        ('faith', 0.2740465066808867, 2),
        ('violation', 0.2684759590811308, 2),
        ('atoms', 0.26070471309222654, 2),
        ('quran', 0.25675524671763944, 2),
        ('study', 0.25272951682516337, 2),
        ('military', 0.24862523031214057, 2),
        ('objectively', 0.24443997539102158, 2),
        ('rz', 0.24017121192015264, 2),
        ('mccullou', 0.23581626062627792, 2),
        ('sentence', 0.23137229105342272, 2),
        ('discuss', 0.22683630804221355, 2),
        ('species', 0.22220513650628743, 2),
        ('flame', 0.2174754042264421, 2),
        ('finds', 0.21264352232628214, 2),
        ('jewish', 0.20770566302224788, 2),
        ('meant', 0.20265773415209595, 2),
        ('snake', 0.19749534987373746, 2),
        ('minority', 0.19221379678362482, 2),
        ('hate', 0.18395131944787158, 2),
        ('miller', 0.1811644042175813, 2),
        ('met', 0.1783434506707047, 2),
        ('interpret', 0.17548765648829678, 2),
        ('eternal', 0.17259618299309748, 2),
        ('association', 0.1696681525257958, 2),
        ('observations', 0.16670264555826533, 2),
        ('ancient', 0.1636986975097882, 2),
        ('genesis', 0.16065529522687585, 2),
        ('hitler', 0.1575713730808523, 2),
        ('bnr', 0.15444580862965546, 2),
        ('contradict', 0.15127741778106205, 2),
        ('psuvm', 0.14806494938335782, 2),
        ('proven', 0.14480707915592253, 2),
        ('handed', 0.14150240285564875, 2),
        ('push', 0.13814942855482498, 2),
        ('whoever', 0.13474656788106432, 2),
        ('women', 0.1312921260387505, 2),
        ('logically', 0.12778429039259698, 2),
        ('contradicts', 0.12422111734494456, 2),
        ('definitions', 0.12060051717635206, 2),
        ('believing', 0.1169202364396637, 2),
        ('fuller', 0.11317783739548676, 2),
        ('aspect', 0.10937067384408097, 2),
        ('motto', 0.10549586253431771, 2),
        ('philosophers', 0.1015502490995454, 2),
        ('heaven', 0.09753036716160812, 2),
        ('jsn', 0.09343238882770144, 2),
        ('fellow', 0.08925206423701829, 2),
        ('sole', 0.08498464703363948, 2),
        ('influence', 0.0806248015619045, 2),
        ('greater', 0.0761664860816768, 2),
        ('halat', 0.07160280423459672, 2),
        ('aided', 0.061890666699547674, 2),
        ('presents', 0.061974683205303936, 2),
        ('mohammad', 0.062058949203985095, 2),
        ('risk', 0.062143465845606254, 2),
        ('jack', 0.06222823428738461, 2),
        ('worse', 0.06738080878070735, 2),
        ('saves', 0.057337514769715395, 2),
        ('updates', 0.057416527702385006, 2),
        ('affect', 0.057495778526359856, 2),
        ('bringing', 0.057575268352913045, 2),
        ('consists', 0.057654998300368646, 2),
        ('risc', 0.0577349694941599, 2),
        ('vast', 0.05781518306688493, 2),
        ('school', 0.05789564015836583, 2)
    ]
    
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
        wordset.add(filteredWord)

    docList = numpy.unique(rows)

    DOC_ID_TO_LABEL_MAP = {}
    for idx, label in enumerate(open('trainLabel.txt', 'r')):
        DOC_ID_TO_LABEL_MAP[idx + 1] = int(label)

    root = decisionTreeLearner(wordset, docList)
