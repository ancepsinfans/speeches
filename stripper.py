from datetime import datetime
from collections import Counter
from statistics import mean
import xlrd
import csv
import nltk
import pandas as pd
import numpy as np

if __name__ != '__main__':
    print("don't forget to download punkt and averaged_perceptron_tagger")
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')

# makes dictionary from csv/txt
def webster(file, type=1):
    if type == 1:
        dic = dict(csv.reader(open(file, 'r')))
    else:
        dic = dict([line.split() for line in open(file).read().replace('\t', ' ').split('\n') if len(line) != 0])
    for key in dic:
        dic[key] = float(dic[key])
    return dic

# subsection - dictionaries
valence = webster('valence.csv', type=1)
arousal = webster('arousal.csv', type=1)
dominance = webster('dominance.csv', type=1)
anger = webster('anger-scores.txt', type=0)
sadness = webster('sadness-scores.txt', type=0)
anticipation = webster('anticipation-scores.txt', type=0)
disgust = webster('disgust-scores.txt', type=0)
fear = webster('fear-scores.txt', type=0)
joy = webster('joy-scores.txt', type=0)
surprise = webster('surprise-scores.txt', type=0)
trust = webster('trust-scores.txt', type=0)
humor = pd.read_csv('humor_dataset.csv')
humor_tot = dict(zip(humor['word'], humor['mean']))
humor_M = dict(zip(humor['word'], humor['mean_M']))
humor_F = dict(zip(humor['word'], humor['mean_F']))
humor_young = dict(zip(humor['word'], humor['mean_young']))
humor_old = dict(zip(humor['word'], humor['mean_old']))
selfish = ['I’m', "i'll", 'i’ll', 'me', 'I’ll', 'i’d', "i'd", 'my', 'myself', 'i’m', "i'm", 'i’ve', "i've", 'I', 'i’d’ve', "i'd've", 'i', 'mine']


# create local / global averages
def emoav(emo, selfemo, wrdct):
    emo_len = len(selfemo)
    perc_emo = emo_len / wrdct
    scores = [emo[word] for word in selfemo]
    if len(scores) == 0:
      return 0, 0
    else:
      avg_loc = mean(scores)
      avg_glo = avg_loc * perc_emo
      return avg_loc, avg_glo

# opens text file from excel
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

# feeds opened file into class
def hostess(data):
    return [Stripper(data[i]['text'], data[i]['index'], data[i]['date'], data[i]['time'], data[i]['person'], data[i]['diagnosis'], data[i]['age'], data[i]['gender']) for i in range(len(data))]

class Stripper:
    """For getting basic info from a file"""
    def __init__(self, text, index, date, time, person, diagnosis, age, gender):
        # imported info from excel
        self.stripped = text
        self.index = index
        self.date = datetime.strptime(date, '%d/%m/%y')
        self.time = time
        self.person = person
        self.diagnosis = diagnosis       
        self.age = age
        self.gender = gender       

        # top-line features
        self.sents = nltk.tokenize.sent_tokenize(self.stripped)
        self.pos_sents = [nltk.pos_tag(self.sents[i].split(' ')) for i in range(len(self.sents))]
        self.num_sents = len(self.sents)
        self.words_tok = nltk.tokenize.word_tokenize(self.stripped)
        self.words_raw = [x for x in self.stripped.split(' ')]
        self.wordcount = len(self.words_raw)
        self.clean_words = [word.lower().replace('.', '').replace(',', '') for word in self.words_raw]
        self.unique_words = set(self.clean_words)
        
        # POS features
        self.pos = nltk.pos_tag(self.words_tok)
        self.counts = Counter(tag for word,tag in self.pos)
        r = 0
        for key in self.counts:
            r += self.counts[key]
        r = r - (self.counts[','] + self.counts['.'])
        # all individual counts are relative to *pos counts* (not wordcounts)
        self.noun = 100 * (self.counts['NN'] + self.counts['NNS'] + self.counts['NNP'] + self.counts['NNPS']) / r
        self.pron = 100 * (self.counts['PRP'] + self.counts['PRP$']) / r
        self.adj = 100 * (self.counts['JJ'] + self.counts['JJR'] + self.counts['JJS']) / r
        self.adv = 100 * (self.counts['RB'] + self.counts['RBR'] + self.counts['RBS']) / r
        self.intj = 100 * (self.counts['UH']) / r
        self.verb = 100 * (self.counts['VB'] + self.counts['VBD'] + self.counts['VBG'] + self.counts['VBN'] + self.counts['VBP'] + self.counts['VBZ']) / r
        self.wh = 100 * (self.counts['WDT'] + self.counts['WP'] + self.counts['WP$'] + self.counts['WRB']) / r
        self.conj = 100 * (self.counts['CC']) / r
        self.prep = 100 * (self.counts['IN'] + self.counts['TO']) / r
        
        # fragments
        self.frag = 0
        self.sent_counts = [Counter(tag for word, tag in sent) for sent in self.pos_sents]
        verbs = {'VB','VBD','VBG','VBN','VBP','VBZ'}
        for sent in self.sent_counts:
            if len(verbs-set(sent.keys())) == len(verbs):
                self.frag += 1
                
        # word frequencies
        self.freq = nltk.FreqDist(word for word in self.clean_words)
        func = open('funcwords.txt').read().translate(str.maketrans("',[]", '    ')).split()
        self.funcs_freq = 100 * sum(self.freq[word] for word in self.freq if word in func) / self.wordcount
        self.rare = [word for word in self.freq if self.freq[word] == 1]
        self.rare_nonfunc = [word for word in self.rare if word not in func]
        self.selfish = sum(self.freq[word] for word in self.freq if word in selfish) / self.num_sents
        
        self.anger = [word for word in self.clean_words if word in anger.keys()]
        self.sadness = [word for word in self.clean_words if word in sadness.keys()]
        self.anticipation = [word for word in self.clean_words if word in anticipation.keys()]
        self.disgust = [word for word in self.clean_words if word in disgust.keys()]
        self.fear = [word for word in self.clean_words if word in fear.keys()]
        self.joy = [word for word in self.clean_words if word in joy.keys()]
        self.surprise = [word for word in self.clean_words if word in surprise.keys()]
        self.trust = [word for word in self.clean_words if word in trust.keys()]
        
        self.valence = [word for word in self.clean_words if word in valence.keys()]
        self.arousal = [word for word in self.clean_words if word in arousal.keys()]
        self.dominance = [word for word in self.clean_words if word in dominance.keys()]

        self.humor = [word for word in self.clean_words if word in humor_tot.keys()]
        
        # emotional strength
        self.al_fear, self.ag_fear = emoav(fear, self.fear, self.wordcount)
        self.al_joy, self.ag_joy = emoav(joy, self.joy, self.wordcount)
        self.al_trust, self.ag_trust = emoav(trust, self.trust, self.wordcount)
        self.al_surprise, self.ag_surprise = emoav(surprise, self.surprise, self.wordcount)
        self.al_disgust, self.ag_disgust = emoav(disgust, self.disgust, self.wordcount)
        self.al_anticipation, self.ag_anticipation = emoav(anticipation, self.anticipation, self.wordcount)
        self.al_anger, self.ag_anger = emoav(anger, self.anger, self.wordcount)
        self.al_sadness, self.ag_sadness = emoav(sadness, self.sadness, self.wordcount)

        self.al_valence, self.ag_valence = emoav(valence, self.valence, self.wordcount)
        self.al_arousal, self.ag_arousal = emoav(arousal, self.arousal, self.wordcount)
        self.al_dominance, self.ag_dominance = emoav(dominance, self.dominance, self.wordcount)

        self.al_humor_tot, self.ag_humor_tot = emoav(humor_tot, self.humor, self.wordcount)
        self.al_humor_M, self.ag_humor_M = emoav(humor_M, self.humor, self.wordcount)
        self.al_humor_F, self.ag_humor_F = emoav(humor_F, self.humor, self.wordcount)
        self.al_humor_young, self.ag_humor_young = emoav(humor_young, self.humor, self.wordcount)
        self.al_humor_old, self.ag_humor_old = emoav(humor_old, self.humor, self.wordcount)


       
        # sentence and word size
        self.short_sent, self.long_sent = 0, 0
        for sent in self.sents:
            if len(sent.split(' ')) <= 6:
                self.short_sent += 1
            else:
                self.long_sent += 1

        self.short_word, self.long_word = 0, 0
        for word in self.clean_words:
            if len(word) <= 6:
                self.short_word += 1
            else:
                self.long_word += 1

    
    # either gives brief description or returns numpy array
    def bare(self, show=False):
        if show == True:
            print('index: {} / date: {}'.format(self.index, self.date.strftime('%d %b %Y')))
            print('# of sentences: {} / # words: {}'.format(self.num_sents, self.wordcount))
            print('# nouns: {} / # verbs: {} / # adj: {} / # pron: {}'.format(self.noun,self.verb,self.adj,self.pron))
        else:
            return np.array([self.index, self.person, self.diagnosis, self.age, self.gender, self.time, self.num_sents, self.wordcount, self.noun, self.verb, self.adj, self.adv, self.intj, self.pron, self.wh, self.conj, self.prep, self.short_sent, self.long_sent, self.short_word, self.long_word, self.frag, len(self.unique_words), self.funcs_freq, self.selfish, (100*len(self.rare_nonfunc)/self.wordcount), (100*len(self.joy)/self.wordcount), (100*len(self.surprise)/self.wordcount), (100*len(self.anticipation)/self.wordcount), (100*len(self.anger)/self.wordcount), (100*len(self.fear)/self.wordcount), (100*len(self.trust)/self.wordcount), (100*len(self.disgust)/self.wordcount), (100*len(self.sadness)/self.wordcount), self.al_fear, self.ag_fear, self.al_joy, self.ag_joy, self.al_trust, self.ag_trust, self.al_surprise, self.ag_surprise, self.al_disgust, self.ag_disgust, self.al_anticipation, self.ag_anticipation, self.al_anger, self.ag_anger, self.al_sadness, self.ag_sadness, self.al_valence, self.ag_valence, self.al_arousal, self.ag_arousal, self.al_dominance, self.ag_dominance, self.al_humor_tot, self.ag_humor_tot, self.al_humor_M, self.ag_humor_M, self.al_humor_F, self.ag_humor_F, self.al_humor_young, self.ag_humor_young, self.al_humor_old, self.ag_humor_old, (100*len(self.humor)/self.wordcount), (100*len(self.valence)/self.wordcount)])
        
# converts arrays into dataframe        
def cashout(arrays):
    temp = [[arrays[i].bare()] for i in range(len(arrays))]
    return pd.DataFrame(np.concatenate(temp), columns=['index', 'person', 'diagnosis', 'age', 'gender', 'time','sents','wordcount','noun','verb','adj','adv','intj','pron','wh','conj','prep','short sent','long sent','short word','long word','frag','# unique','% func words', '% selfish', '% unique no func', '% joy', '% surprise', '% anticipation', '% anger', '% fear', '% trust', '% disgust', '% sadness', 'loc avg fear', 'glb avg fear', 'loc avg joy', 'glb avg joy', 'loc avg trust', 'glb avg trust', 'loc avg surprise', 'glb avg surprise', 'loc avg disgust', 'glb avg disgust', 'loc avg anticipation', 'glb avg anticipation', 'loc avg anger', 'glb avg anger', 'loc avg sadness', 'glb avg sadness', 'loc avg valence', 'glb avg valence', 'loc avg arousal', 'glb avg arousal', 'loc avg dominance', 'glb avg dominance', 'loc avg humor tot', 'glb avg humor tot', 'loc avg humor M', 'glb avg humor M', 'loc avg humor F', 'glb avg humor F', 'loc avg humor young', 'glb avg humor young', 'loc avg humor old', 'glb avg humor old', '% humor', '% VAD'])

# calculates composite measures into the dataframe
def dance(ndf):
    ndf['short/long'] = ndf['short sent'] / ndf['long sent']
    ndf['frag/min'] = ndf['frag'] / ndf['time'] * 60
    ndf['words/min'] = ndf['wordcount'] / ndf['time'] * 60
    ndf['words/sent'] = ndf['wordcount'] / ndf['sents']
    ndf['verb/noun'] = ndf['verb'] / ndf['noun']
    ndf['noun/pron'] = ndf['noun'] / ndf['pron']
    ndf['% unique w/func'] = ndf['# unique'] / ndf['wordcount'] * 100
    ndf['misc words'] = ndf['wordcount'] - (ndf['noun'] + ndf['verb'] + ndf['adj'] + ndf['adv'] + ndf['pron'] + ndf['intj'] + ndf['wh'] + ndf['prep'] + ndf['conj'])
    ndf['% frag'] = ndf['frag'] / ndf['sents'] * 100
    
# does all of the above in one line with the option of switching to manipulate a single object
def VIP(file, xxx=False):
    if xxx == False:
        temp = cashout(hostess(bouncer(file)))
        dance(temp)
        return temp
    else:
        return hostess(bouncer(file))