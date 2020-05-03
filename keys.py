# coding=utf8
from collections import OrderedDict
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import nltk
import os
from nltk.tag import tnt
from nltk.corpus import indian

def Punctuation(string): 
    punctuations = '''!()-[]{};:'",<>./?@#$%^&*_~|0123456789'''
    for x in string.lower(): 
        if x in punctuations: 
            string = string.replace(x, " ")

    return string

def hindi_model():
    train_data = indian.tagged_sents('hindi.pos')
    tnt_pos_tagger = tnt.TnT()
    tnt_pos_tagger.train(train_data)
    return tnt_pos_tagger

def tagger1(sen):
    return model.tag(nltk.word_tokenize(sen))

nlp = spacy.load('en_core_web_sm')

class TextRank4Keyword():
    """Extract keywords from text"""

    def __init__(self):
        self.d = 0.85 # damping coefficient, usually is .85
        self.min_diff = 1e-5 # convergence threshold
        self.steps = 10 # iteration steps
        self.node_weight = None # save keywords and its weight


    def set_stopwords(self, stopwords):
        """Set stop words"""
        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = nlp.vocab[word]
            lexeme.is_stop = True

    def sentence_segment(self, doc, candidate_pos, lower,bigrams,trigrams):
        """Store those words only in cadidate_pos"""
        sentences = []
        for sent in doc.sents:
            selected_words = []
            bigram_words=[]
            for token in sent:
                bigram_words.append(token.text)
                # Store words only with cadidate POS tag
                if token.pos_ in candidate_pos and token.is_stop is False:
                    if lower is True:
                        selected_words.append(token.text.lower())
                    else:
                        selected_words.append(token.text)
            if bigrams==True:
                for i in range(len(sent)-1):
                    if sent[i].pos_ in candidate_pos and sent[i].is_stop is False and sent[i+1].pos_ in candidate_pos and sent[i+1].is_stop is False:
                        if lower is True:
                            selected_words.append(sent[i].text.lower())
                        else:
                            selected_words.append(str(sent[i].text+" "+sent[i+1].text))
            if trigrams==True:
                for i in range(len(sent)-2):
                    if sent[i].pos_ in candidate_pos and sent[i].is_stop is False and sent[i+1].pos_ in candidate_pos and sent[i+1].is_stop is False and sent[i+2].pos_ in candidate_pos and sent[i+2].is_stop is False:
                        if lower is True:
                            selected_words.append(sent[i].text.lower())
                        else:
                            selected_words.append(str(sent[i].text+" "+sent[i+1].text+" "+sent[i+2].text))
            sentences.append(selected_words)
        return sentences

    def get_vocab(self, sentences):
        """Get all tokens"""
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab

    def get_token_pairs(self, window_size, sentences):
        """Build token_pairs from windows in sentences"""
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i+1, i+window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs

    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())

    def get_matrix(self, vocab, token_pairs):
        """Get normalized matrix"""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1

        # Get Symmeric matrix
        g = self.symmetrize(g)

        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm!=0) # this is ignore the 0 element in norm

        return g_norm


    def get_keywords(self, number=10):
        """Print top number keywords"""
        # print(len(self.node_weight.items()))
        node_weight_list = [list(ele) for ele in self.node_weight.items()]
        for i in node_weight_list:
            res = len(i[0].split())
            if res==2:
                # weight for bigrams
                i[1]=4*i[1]
            if res==3:
                # weight for trigrams
                i[1]=6*i[1]

        node_weight = OrderedDict(sorted(node_weight_list, key=lambda t: t[1], reverse=True))
        for i, (key, value) in enumerate(node_weight.items()):
            # print(key + ' - ' + str(value))
            print(i+':'+key)
            if i > number:
                break


    def analyze(self, text,
                candidate_pos=['NOUN', 'VERB'],
                window_size=4, lower=False,bigrams=False,trigrams=False, stopwords=list()):
        """Main function to analyze text"""

        # Set stop words
        self.set_stopwords(stopwords)

        # Pare text by spaCy
        doc = nlp(text)

        #
        # Filter sentences
        sentences = self.sentence_segment(doc, candidate_pos, lower,bigrams,trigrams) # list of list of words

        # ences)
        # Build vocabulary
        vocab = self.get_vocab(sentences)


        # Get token_pairs from windows
        token_pairs = self.get_token_pairs(window_size, sentences)


        # Get normalized matrix
        g = self.get_matrix(vocab, token_pairs)


        # Initionlization for weight(pagerank value)
        pr = np.array([1] * len(vocab))


        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1-self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr))  < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]

        self.node_weight = node_weight

#Uncomment Below to use get pos tags, filter only NN,NNC,Ukn, verbs
#Way slower and not better
# model = hindi_model()
# input = open("data/physics_text.txt","r")
# inp = input.read()
# sentences = inp.split("\n")
# new_tagged = map(tagger1,sentences)


# file = open("data/no_stops.txt", "w")
# file.writelines( list( "%s\n" % item for item in new_tagged ) )
# file.close()
# # print(list(new_tagged))

with open('data/stopwords.txt', 'r') as myfile:
  stoplist = myfile.read()
stoplist = stoplist.split(' ')

with open('data/history_text.txt','r') as hin_data:#replace with whatever corpus
    data = hin_data.read()

hinwords = data.split()
resultwords  = [word for word in hinwords if word not in stoplist and len(word) > 4] # way faster than using postags
result = ' '.join(resultwords)


text_file = open("data/no_stops.txt", "w")
text_file.write(Punctuation(result))
text_file.close()

with open('data/no_stops.txt','r') as f:
    text = f.read()

tr4w = TextRank4Keyword()
tr4w.analyze(text, candidate_pos = ['NOUN', 'VERB'], window_size=4, lower=False,bigrams=True,trigrams=True)#set bigram,trigram flags false to process quicker
tr4w.get_keywords(20)
