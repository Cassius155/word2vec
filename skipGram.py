import numpy as np
import re

class skipGram:
    def __init__(self, textPath, windowSize, negSampleRate, embeddingSize):
        self.tokenized = self.getText(textPath)
        self.windowSize = windowSize
        self.negSampleRate = negSampleRate
        self.embeddingSize = embeddingSize

        self.vocabSize = None
        self.corpus = None
        self.wordfreqs = None

    def getText(self, path):
        with open(path, encoding="utf-8") as f:
            textSep = f.readlines()
        f.close()
        self.corpus =  " ".join(textSep)
        clean = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
        return clean.findall(self.corpus.lower())
    
        
    def maptoIndex(self, text):
        """
        Creates a mapping from every unique word in text to an index
        Input: text: List of strings corresponding to training data
        Output: Dictionary containing word to index pairs
        """
        word2idx = {}
        idx = 0
        for word in text:
            if word not in word2idx.keys():
                word2idx[word] = idx
                idx+=1
        return word2idx

    def unigramFreq(self, textIdxs, power = 0.75):
        """
        Calculates the unigram distribution raised to the 3/4 power
        Input:  textIdxs: input training text expressed using the index for each word
                power: power to raise unigram distribution to, default is 3/4
        Output: List of the unigram distribution probabilities
        """
        frq = np.zeros(self.vocabSize)
        for i in textIdxs:
            frq[i] += 1
        frq = frq ** power
        return frq/frq.sum()

    def lossFunc(self, word, context, negatives):
        """
        Returns calculated loss for given embedings
        Input:  word: vector embedding of center word
                context: vector embedding of context word for center word
                negatives: vector embedings of k negative samples, with k = negSampleRate
        Output: Returns calculated loss according to the negative sampling objective
        """
        return -np.log(self.sigmoid(np.dot(context, word))) + np.sum(-np.log(self.sigmoid(-np.dot(negatives, word))))
    
    def textToIndexes(self, text, mapping):
        """
        Converts list of words to list of indexes based on mapping
        Input:  list of words, corresponding to text
                mapping of words to indexes
        Output: list of indexes for each word in text, if no mapping found -1 is used
        """
        idxText = []
        for word in text:
            idxText.append(mapping.get(word, -1))
        return idxText

    def genPositiveSamples(self, textAsIndexes):
        """
        Generates positive samples for every word in corpus based on windowSize
        Input:  textAsIndices: input training text expressed using the index for each word
        Output: list of lists containing positive samples for every word in vocabulary
        """
        #Recreate text using word indices and count frequencies
        self.wordfreqs = self.unigramFreq(textAsIndexes)

        positiveSamples = [None]*self.vocabSize
        for idx, word in enumerate(textAsIndexes):
            if(word == -1):
                continue
            start = max(0, idx-self.windowSize)
            stop = min(len(textAsIndexes), idx+self.windowSize+1)
            sampleInstance = []
            for i in range(start, stop):
                if(i == idx):
                    continue
                else:
                    sampleInstance.append(textAsIndexes[i])
            if(positiveSamples[word] == None):
                positiveSamples[word] = sampleInstance
            else:
                positiveSamples[word].extend(sampleInstance)
        return positiveSamples
    
    def genNegativeSamples(self, word, posSamples):
        """
        Generates negSamplerate number of negative samples for the given word, according to unigram distribution
        Input:  word: index of center word to generate negative samples for
                posSamples: indexes of positive samples for the center word
        Output: list of indexes of negative samples
        """
        #Generate probabilities excluding positive samples and center word
        localFreqs = np.copy(self.wordfreqs)
        localFreqs[word] = 0
        for idx in posSamples:
            localFreqs[idx] = 0
        localFreqs /= localFreqs.sum()
        #Sample based on negSampleRate
        return np.random.choice(self.vocabSize, size=self.negSampleRate, p=localFreqs)
    
    def sigmoid(self, x):
        """
        Applies sigmoid function to specified scalar or vector
        Input: x: scalar or vector
        Output: sigmoid function applied to x
        """
        return 1.0/(1.0+np.exp(-x))
    
    def initWeights(self):
        """
        Initialises the weight matrices with random values
        Output: W1: first layer weight matrix
                W2: second layer weight matrix
        """
        W1 = np.random.rand(self.embeddingSize, self.vocabSize)
        W2 = np.random.rand(self.vocabSize, self.embeddingSize)
        return W1, W2
    
    def getGradients(self, center, context, negatives):
        """
        Return the gradients for one instance of word-context-negative samples
        Input:  center: embedding of center word
                context: embedding of context word
                negatives: embeddings of k negative samples, k = negSampleRate
        Output: centerGradient: gradient of loss function with respect to center word embedding
                contextGradient: gradient of loss function with respect to context word embedding
                negativeGradients: gradient of loss function with respect to negative sample embeddings
        """
        wordContextSim = np.dot(center, context)
        wordNegativesSim = negatives @ center

        wordContextSigmoid = self.sigmoid(wordContextSim)
        wordNegativeSigmoid = self.sigmoid(wordNegativesSim)

        contextGradient = (wordContextSigmoid - 1) * center
        negativesGradients = wordNegativeSigmoid[:, None] * center
        centerGradient = (wordContextSigmoid - 1) * context + wordNegativeSigmoid @ negatives

        return centerGradient, contextGradient, negativesGradients
    
    def train(self, learningRate, epochs):
        """
        Trains the model with specified learning rate for specified epochs
        Input:  learningRate: learning rate of model
                epochs: number of epochs to train model for
        Output: W1: Input embedding matrix
                W2: Output embedding matrix
                losses: value of loss function after each epoch
                word2idx: dictionary with word to index pairs
        """
        #Clean data and return as list of words
        cleanText = self.tokenized

        #Create mapping of unique words in corpus to index
        word2idx = self.maptoIndex(cleanText)
        self.vocabSize = len(word2idx)

        W1, W2 = self.initWeights()
        indexText = self.textToIndexes(cleanText, word2idx)

        #Generate positive pairs for each word in corpus
        positivePairs = self.genPositiveSamples(indexText)

        losses = []

        for epoch in range(epochs):
            epochLoss = 0
            for word in range(self.vocabSize):
                centerEmbedding = W1[:, word]
                posSamples = positivePairs[word]
                for pos in posSamples:
                    negSamples = self.genNegativeSamples(word, posSamples)

                    contextEmbedding = W2[pos, :]
                    negativesEmbedding = W2[negSamples, :]

                    centerGrad, contextGrad, negGrads = self.getGradients(centerEmbedding, contextEmbedding, negativesEmbedding)

                    #Update weight matrices
                    W1[:, word] -= learningRate*centerGrad
                    W2[pos, :] -= learningRate*contextGrad
                    W2[negSamples, :] -= learningRate*negGrads

                    loss = self.lossFunc(W1[:, word], W2[pos, :], W2[negSamples, :])

                    epochLoss += loss
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {epochLoss:.4f}")
            losses.append(epochLoss)
        return W1, W2, losses, word2idx

if __name__ == "__main__":
    net = skipGram("trainingData/inputText.txt", windowSize=5, negSampleRate=15, embeddingSize=100)
    net.train(0.05, 300)