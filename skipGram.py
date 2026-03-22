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
    
        
    def maptoIndex(text):
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

    def unigramFreq(textIdxs, size, power = 0.75):

        frq = np.zeros(size)
        for i in textIdxs:
            frq[i] += 1
        frq = frq ** power
        return frq/frq.sum()

    def lossFunc(self, word, context, negatives):
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
        #Recreate text using word indices and count frequencies
        self.wordfreqs = self.unigramFreq(textAsIndexes, self.vocabSize)

        positiveSamples = []
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
            positiveSamples.append(sampleInstance)
        return positiveSamples
    
    def genNegativeSamples(self, word, posSamples):
        #Generate probabilities excluding positive samples and 
        localFreqs = np.copy(self.wordfreqs)
        localFreqs[word] = 0
        for idx in posSamples:
            localFreqs[idx] = 0
        localFreqs /= localFreqs.sum()
        #Sample based on negSampleRate
        return np.random.choice(self.vocabSize, size=self.negSampleRate, p=localFreqs)
    
    def sigmoid(self, x):
        return 1.0/(1.0+np.exp(-x))
    
    def initWeights(self):
        W1 = np.random.rand(self.embeddingSize, self.vocabSize)
        W2 = np.random.rand(self.vocabSize, self.embeddingSize)
        return W1, W2
    
    def getGradients(self, center, context, negatives):
        """
        Return the gradients for one instance of word-context-negative samples
        Input:  center: embedding of center word
                context: embedding of context word
                negatives: embeddings of k negative samples, k = negSampleRate
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
        #Clean data and return as list of words
        cleanText = self.tokenized

        #Create mapping of unique words in corpus to index
        word2idx = map(cleanText)
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
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {epochLoss:.4f} | Pairs: ")
            losses.append(epochLoss)
        return W1, W2, losses, word2idx

if __name__ == "__main__":
    net = skipGram("inputText.txt", 5, 5, 100)
    net.train(0.01, 100)