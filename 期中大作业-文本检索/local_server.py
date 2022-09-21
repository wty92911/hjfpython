import re
import json
import math
import socket
import string
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from threading import Thread
from nntplib import ArticleInfo
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn import datasets, preprocessing
from tqdm import tqdm
def wrap(data):
    return bytes(json.dumps(data).encode('utf-8'))
class LocalServer(object):
    def ccut(self,x):
        '''正则化、去除无关'''
        for i in range(1,2):
            x[i] = re.sub('[^A-Za-z]+', ' ', x[i]).lower()
            x[i] = x[i].translate(self.remove)
            x[i] = word_tokenize(x[i])
            ls = []
            for w in x[i]:
                if w not in self.stop_words:
                    ls.append(self.lemmatizer.lemmatize(w))
            x[i] = ls
        return x
     def getdt(self,x):
        '''get word count dt'''
        for i in range(1,2):
            for word in x[i]:
                if word in self.dt:
                    self.dt[word] += 1
                else:
                    self.dt[word] = 1
        return x
    def getsimilarword(self,w):
        '''在词典中找到相似词'''
        ls = []
        for x in self.dt.keys():
            if fuzz.ratio(x,w) >= 80:
                ls.append(x)
        return ls
   
    def getset(self,x):
        '''make body to a set'''
        return set(x[1]) | set(x[0])
    def getTF_IDF(self,x):
        '''get tf-idf'''
        dt = {}
        for word in x[1]:
            if word in dt:
                dt[word] += 1
            else:
                dt[word] = 1
        ls = []
        for word in list(self.dt.keys()):
            if word in dt:
                ls.append(dt[word] / len(x[1]) * self.dt[word])
            else:
                ls.append(0)
        a = np.array(ls).astype('float')
        a = a.reshape((1,len(ls)))
       # print(a.shape)
        return a
    def getidf(self,x):
        '''get idf for client terms'''
        cnt = 0
        for s in self.data['setwords']:
            if x in s:
                cnt += 1
        return math.log(len(self.data['setwords']) / cnt)
    def writevocab(self):
        with open('vocab.txt','w') as f:
            for w in self.dt.keys():
                f.write("{}\n".format(w))
    def writesimilar(self):
        with open('synonym.txt','w') as f:
            ls = list(self.dt.keys())
            for i in tqdm(range(len(ls))):
                f.write('{} : {}\n'.format(ls[i],self.getsimilarword(ls[i])))
    def __init__(self, host, port):
        print("local server is initing...")
        self.address = (host, port)
        self.data = pd.read_csv('data/all_news.csv')
        self.doc = pd.read_csv('data/all_news.csv')
        self.stop_words = set(stopwords.words('english'))
        self.remove = str.maketrans('','',string.punctuation)
        self.lemmatizer = WordNetLemmatizer()
        self.data = self.data.apply(self.ccut,axis = 1)
        self.dt = {}
        self.data = self.data.apply(self.getdt,axis = 1)
        for x in list(self.dt.keys()):
            if self.dt[x] < 20:
                del self.dt[x]
        print('dict.length is {}'.format(len(self.dt)))
        self.data['setwords'] = self.data.apply(self.getset,axis = 1)
        self.pos = {}
        ls = list(self.dt.keys())
        for i in range(len(ls)):
            self.pos[ls[i]] = i
            self.dt[ls[i]] = self.getidf(ls[i])
            #print(self.dt[x])
        self.data['TF-IDFvec'] = self.data.apply(self.getTF_IDF,axis = 1)
        self.writevocab()
        self.writesimilar()
        #self.data.to_csv('cut.csv')
    def cos(self,a,b):
        '''计算向量cos'''
        x = a.dot(b.T)
        y = np.linalg.norm(a) * np.linalg.norm(b)
        if(y == 0):
            return 0
        return x / y
    def evaluate(self):
        '''实现聚类，计算purity'''
        print('local server is evaluating...')
        #print(self.data['TF-IDFvec'].shape)
        vec = np.hstack(self.data['TF-IDFvec']).reshape((len(self.data),len(self.dt)))
        vec = preprocessing.scale(vec)
        km = KMeans(n_clusters=5,random_state=666).fit(vec)
        pred = km.predict(vec)
        orig = []
        s = set()
        for i in range(len(vec)):
            s.add(self.data['topic'][i])
            orig.append(len(s) - 1)
        g = np.zeros((5,5))
        for i in range(len(pred)):
            g[pred[i]][orig[i]] += 1
        purity = np.sum([np.max(g[i]) for i in range(len(g))]) / len(pred)
        print("the purity is {}".format(purity))
    def include(self,data,terms):
        '''判断文章x是否含有搜索的关键字以及相似词'''
        for w in terms:
            if w in data['setwords']:
                return True
        return False
    def find(self,terms):
        '''先找到包含关键字以及关键字相似词的文章，然后根据tfidf向量计算cos，找到前10篇文章'''
        similarterms = []
        for w in terms:
            print("x is {} sim is \n".format(w))
            print(self.getsimilarword(w))
            similarterms += self.getsimilarword(w)
        article = []
        for i in range(len(self.data)):
            if self.include(self.data.iloc[i],similarterms):
                article.append(i)
        ls = []
        v = np.zeros(len(self.dt))
        for w in terms:
            if w in self.pos:
                v[self.pos[w]] = (terms.count(w)) / len(terms) * self.getidf(w)
        for i in article:
            ls.append((i,self.cos(self.data['TF-IDFvec'][i],v)))
        ls.sort(key = lambda x : -x[1])
        ret = []
        for x in ls[:10]:
            ret.append((self.doc['title'][x[0]],self.doc['body'][x[0]]))
        return ret
    def serve(self,new_socket,client_info):
        '''处理客户端发送来的数据'''
        print("client{} is connected".format(client_info))
        raw_data = new_socket.recv(1024)
        while raw_data:
            data = json.loads(raw_data)
            print('data is {}'.format(data))
            new_socket.send(wrap(self.find(data)))
            raw_data = new_socket.recv(1024)
        new_socket.close()
        print("client{} is unconnected".format(client_info))
    
    def run(self):
        print('local server is running...')
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(self.address)
        server.listen(10)
        while True:
            new_socket,client_info = server.accept()
            p = Thread(target = self.serve,args = (new_socket,client_info))
            p.start()
        
server = LocalServer("127.0.0.1", 9001)
server.evaluate()
server.run()