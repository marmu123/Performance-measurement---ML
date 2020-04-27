import math
def predictionErrorMultiTarget(realOutput,computedOutput):
    '''

   input: realOutputss, computedOutputss - k-dimensional arrays of the same length
    containing real values (two matrix of k \times noSamples$ elements, $k$ = number of output targets,
    noSamples = no of samples/examples)

    output: prediction error - real value

    '''
    tot=0
    k=len(realOutput)
    noSamples=len(realOutput[0])
    for i in range(k):
        errAct=sum([(abs(realOutput[i][j]-computedOutput[i][j]))/noSamples for j in range(noSamples)])/noSamples
        tot+=errAct
    tot/=k
    return tot

def testpredictionErrorMultiTarget():


    assert (predictionErrorMultiTarget([[1,1],[2,2]],[[1,1],[2,2]]) == 0)
    assert (predictionErrorMultiTarget([[2,4],[2,4]],[[2,4],[4,8]]) - 0.75 < 0.0001)
    assert (predictionErrorMultiTarget([[1,2],[1,7]],[[10,3],[2,10]]) - 1.75 < 0.0001)



def multiClass(realOutput,computedOutput):
    '''
   input: realLabels, computedLabels - one-dimensional arrays of the same length containing labels
   (two arrays of noSamples labels from {label_1, label_2, ..., label_C},
    noSamples = no of samples/exampeles)
    output: prediction quality expressed by accuracy, precison and recall.
    '''
    labels=set(realOutput)
    acc=sum([1 if realOutput[i]==computedOutput[i] else 0  for i in range(len(computedOutput))])
    prec={}
    recall={}
    for el in labels:
        tp=sum([1 if realOutput[i]==el and computedOutput[i]==el else 0  for i in range(len(computedOutput))])
        fp=sum([1 if realOutput[i]!=el and computedOutput[i]==el else 0  for i in range(len(computedOutput))])
        fn=sum([1 if realOutput[i]==el and computedOutput[i]!=el else 0  for i in range(len(computedOutput))])
        prec[el]=tp/(tp+fp)
        recall[el]=tp/(tp+fn)

    return acc,prec,recall


def testMultiClass():
    real=    ["cat","dog","panda","cat","cat","panda"]
    computed=["cat","dog","cat","cat","dog","panda"]
    acc,prec,recall=multiClass(real,computed)
    assert acc==4
    precRight={'cat':2/3,'dog':1/2,'panda':1}
    recallRight={'cat':2/3,'dog':1,'panda':0.5}
    for key in prec:
        assert abs(prec[key]-precRight[key])<0.0001

    for key in recall:
        assert abs(recall[key] - recallRight[key]) < 0.0001





def lossForRegression(realOutput,computedOutput):
    return predictionErrorMultiTarget(realOutput,computedOutput)


def lossForBinaryClassification(realOutput,computedOutput):
    '''
    outputul clasificatorului este reprezentat ca o matrice cu noSamples * 2 valori reale subunitare;
    fiecare linie are suma elementelor 1, elementele reprezentand probabilitatile prezise pt fiecare din cele 2 clase).
    https://gombru.github.io/2018/05/23/cross_entropy_loss/
    Cross entropy loss
    '''
    rez=0
    err=[]
    for i,el in enumerate(computedOutput):
        bestMatchComputed=max(el)
        bestIndexComputed=el.index(bestMatchComputed)
        bestMatchReal=max(realOutput[i])
        bestIndexReal=realOutput[i].index(bestMatchReal)
        if bestIndexReal==bestIndexComputed:
            pr=1
        else:
            pr=0
            err+=[computedOutput[i]]
        rez+=-1*(pr*math.log2(bestMatchComputed)+(1-pr)*math.log2(1-bestMatchComputed))
    return rez,err


def testLossForBinaryClassification():
    print(lossForBinaryClassification([[0.9,0.1],[0.7,0.3],[0.2,0.8]],[[0.8,0.2],[0.4,0.6],[0.2,0.8]]))
    print(lossForBinaryClassification([[0.6, 0.4], [0.2, 0.8], [0.45, 0.55]], [[0.4, 0.6], [0.6, 0.4], [0.95, 0.05]]))




testpredictionErrorMultiTarget()
testMultiClass()
testLossForBinaryClassification()