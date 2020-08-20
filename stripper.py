from datetime import datetime
from collections import Counter
import xlrd
import nltk
import pandas as pd
import numpy as np

if __name__ != '__main__':
    print("don't forget to download punkt and averaged_perceptron_tagger")
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

def bouncer(file):
    workbook = xlrd.open_workbook(file, on_demand=True)
    worksheet = workbook.sheet_by_index(0)
    first_row = [worksheet.cell_value(0,col) for col in range(worksheet.ncols)]

    data = []
    for row in range(1, worksheet.nrows):
        elm = {}
        for col in range(worksheet.ncols):
            elm[first_row[col]]=worksheet.cell_value(row,col)
        data.append(elm)
    return data

def hostess(data):
    return [Stripper(data[i]['text'], data[i]['index'], data[i]['date'], data[i]['time']) for i in range(len(data))]
        
class Stripper:
    """For getting basic info from a file"""
    def __init__(self, text, index, date, time):
        self.stripped = text
        self.index = index
        self.date = datetime.strptime(date, '%d/%m/%y')
        self.time = time
        
        self.sents = nltk.tokenize.sent_tokenize(self.stripped)
        self.pos_sents = [nltk.pos_tag(self.sents[i].split(' ')) for i in range(len(self.sents))]
        self.num_sents = len(self.sents)
        self.words_tok = nltk.tokenize.word_tokenize(self.stripped)
        self.words_raw = [x for x in self.stripped.split(' ')]
        self.wordcount = len(self.words_raw)

        self.pos = nltk.pos_tag(self.words_tok)
        self.counts = Counter(tag for word,tag in self.pos)
        self.noun = self.counts['NN'] + self.counts['NNS'] + self.counts['NNP'] + self.counts['NNPS']
        self.pron = self.counts['PRP'] + self.counts['PRP$']
        self.adj = self.counts['JJ'] + self.counts['JJR'] + self.counts['JJS']
        self.adv = self.counts['RB'] + self.counts['RBR'] + self.counts['RBS']
        self.intj = self.counts['UH']
        self.verb = self.counts['VB'] + self.counts['VBD'] + self.counts['VBG'] + self.counts['VBN'] + self.counts['VBP'] + self.counts['VBZ']
        self.wh = self.counts['WDT'] + self.counts['WP'] + self.counts['WP$'] + self.counts['WRB']
        self.conj = self.counts['CC']
        self.prep = self.counts['IN'] + self.counts['TO']
        
        self.frag = 0
        self.sent_counts = [Counter(tag for word, tag in sent) for sent in self.pos_sents]
        verbs = {'VB','VBD','VBG','VBN','VBP','VBZ'}
        for sent in self.sent_counts:
            if len(verbs-set(sent.keys())) == len(verbs):
                self.frag += 1
                
        self.short_sent, self.long_sent = 0, 0
        for sent in self.sents:
            if len(sent.split(' ')) <= 6:
                self.short_sent += 1
            else:
                self.long_sent += 1
        
    def bare(self, show=False):
        if show == True:
            print('index: {} / date: {}'.format(self.index, self.date.strftime('%d %b %Y')))
            print('# of sentences: {} / # words: {}'.format(self.num_sents, self.wordcount))
            print('# nouns: {} / # verbs: {} / # adj: {} / # pron: {}'.format(self.noun,self.verb,self.adj,self.pron))
        else:
            return np.array([self.index, self.time, self.num_sents, self.wordcount, self.noun, self.verb, self.adj, self.adv, self.intj, self.pron, self.wh, self.conj, self.prep, self.short_sent, self.long_sent, self.frag])
        
def cashout(arrays):
    temp = [[arrays[i].bare()] for i in range(len(arrays))]
    return pd.DataFrame(np.concatenate(temp), columns=['index','time','num_sents','wordcount','noun','verb','adj','adv','intj','pron','wh','conj','prep','short_sent','long_sent','frag'])

def dance(ndf):
    ndf['short/long'] = ndf['short_sent'] / ndf['long_sent']
    ndf['frag/min'] = ndf['frag'] / ndf['time'] * 60
    ndf['words/min'] = ndf['wordcount'] / ndf['time'] * 60
    ndf['words/sent'] = ndf['wordcount'] / ndf['num_sents']
