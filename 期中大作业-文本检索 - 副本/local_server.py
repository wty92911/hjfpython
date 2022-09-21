from threading import Thread
import socket
import json
import pandas as pd
import re
import math
import numpy as np
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fuzzywuzzy import fuzz
from sklearn.cluster import KMeans

class LocalServer(object):
    def remove_punc(self,article):
        """
        去除标点符号
        """
        dicts = {i:' ' for i in punctuation}
        punc_table = str.maketrans(dicts)
        for i in range(2):
            article[i] = re.sub(r"\d+",' ',article[i])
            article[i] = article[i].translate(punc_table)
        return article
    def remove_stop(self,article):
        """
        去除停用词
        """
        en_stops = set(stopwords.words('english'))
        for i in range(2):
            l = article[i].split()
            a = []
            for word in l:
                word = word.lower()
                if word not in en_stops:
                    a.append(word)
            article[i] = a
        return article
    def get_lemmatize(self,article):
        """
        调包对词提取词根
        """
        wnl = WordNetLemmatizer()
        for i in range(2):
            l = []
            for word in article[i]:
                l.append(wnl.lemmatize(word))
            article[i] = l
        return article
    def get_count(self,article):
        """
        记录词的总出现次数
        """
        for i in range(2):
            for word in article[i]:
                if word in self.dict:
                    self.dict[word] = self.dict[word] + 1
                else:
                    self.dict[word] = 1
        return article
    def remove_lowfre(self):
        """
        去除低频词，这里将阈值设置为5
        """
        for x in list(self.dict.keys()):
            if self.dict[x] <= 5:
                del self.dict[x]
        return self.dict
    def write_dict(self):
        """
        整理词典，对词按照字典序进行排序，然后输出
        """
        self.vocab = [x for x in list(self.dict.keys())]
        self.vocab.sort()
        self.word_index = {}
        for i in range(len(self.vocab)):
            self.word_index[self.vocab[i]] = i
        f = open("vocab.txt","w",encoding="utf-8") 
        for x in self.vocab:
            f.write(x + ":" + str(self.dict[x]) + "\n")
        f.close()
    def get_idf(self):
        self.idf = {}
        N = len(self.data['word_set'])
        for word in self.vocab:
            cnt = 0
            for article in self.data['word_set']:
                if word in article:
                    cnt += 1
            self.idf[word] = math.log(N / cnt)
        return self.idf
    def get_tfidf(self,x):
        tf = {}
        for word in x[0]:
            if word in tf:
                tf[word] += 1
            else:
                tf[word] = 1
        for word in x[1]:
            if word in tf:
                tf[word] += 1
            else:
                tf[word] = 1
        l = []
        for word in self.vocab:
            if word in tf:
                l.append(tf[word] / (len(x[0]) + len(x[1])) * self.idf[word])
            else:
                l.append(0)
        return np.array(l)
    def calc_cos(self,v1,v2):
        L = np.linalg.norm(v1) * np.linalg.norm(v2)
        if(L == 0):
            return 0
        D = v1.dot(v2.T)
        return D / L
    def get_similar(self,word):
        similar = []
        for x in self.vocab:
            tmp = fuzz.ratio(x,word)
            if(tmp >= 85):
                similar.append([x,tmp])
        similar.sort(key = lambda x : -x[1])
        return similar[:10]
    def get_synonym(self):
        f = open("synonym.txt","w",encoding="utf-8") 
        self.synonym = {}
        for word in self.vocab:
            self.synonym[word] = self.get_similar(word)
            f.write(word + " : " + str(self.synonym[word]) + "\n")
        f.close()
        return self.synonym
    def calc_purity(self):
        print("purity calculating...")
        KX = np.vstack(self.data['TF-IDF'])
        km = KMeans(n_clusters = 10,random_state = 494539).fit(KX)
        pred = km.predict(KX)
        orig = []
        orig_labels = {}
        for i in range(len(pred)):
            _topic = self.data['topic'][i]
            if _topic in orig_labels:
                orig.append(orig_labels[_topic])
            else:
                orig_labels[_topic] = len(orig_labels)
                orig.append(orig_labels[_topic])
        count = np.zeros((10,5))
        for i in range(len(pred)):
            count[pred[i]][orig[i]] += 1
        M = len(pred)
        P = 0
        for r in range(10):
            P += np.max(count[r])
        P = P / M
        print("Purity is {}".format(P))
        return P
    def __init__(self, host, port):
        self.address = (host, port)
        print("loading the articles...")
        self.articles = pd.read_csv("data/all_news.csv")
        self.data = pd.read_csv("data/all_news.csv")
        self.dict = {}
        print("processing the data...")
        self.data = self.data.apply(self.remove_punc,axis = 1)
        self.data = self.data.apply(self.remove_stop,axis = 1)
        self.data = self.data.apply(self.get_lemmatize,axis = 1)
        print("generating the dictionary...")
        self.data.apply(self.get_count,axis = 1)
        self.remove_lowfre()
        #print(self.dict)
        self.write_dict()
        print("calc the TF-IDF...")
        self.data['word_set'] = self.data.apply(lambda x : set(x[0]) | set(x[1]),axis = 1)
        self.get_idf()
        self.data['TF-IDF'] = self.data.apply(self.get_tfidf,axis = 1)
        print("get_synonym...")
        #self.get_synonym()
        #print(self.data)
        #self.calc_purity()
        print("initialization completed!")
    def search(self,terms):
        articles = []
        tf_idf = np.zeros(len(self.vocab))
        tmpls = []
        result = []
        for i in range(len(self.data)):
            suc = 0
            for word in terms:
                if(word in self.data.iloc[i]['word_set']):
                    suc = 1
            if(suc == 1):
                articles.append(i)
        for word in terms:
            if word in self.vocab:
                tf_idf[self.word_index[word]] = (terms.count(word)) / len(terms) * self.idf[word]
            else:
                s = self.get_similar(word)
                if(len(s)):
                    tmp = s[0][0]
                    tf_idf[self.word_index[get_similar(tmp)]] = (terms.count(tmp)) / len(terms) * self.idf[tmp]
        for article in articles:
            tmpls.append((article,self.calc_cos(tf_idf,self.data['TF-IDF'][article])))
        tmpls.sort(key = lambda x : -x[1])
        for x in tmpls[:min(len(tmpls),10)]:
            id = x[0]
            result.append((self.articles['title'][id],self.articles['body'][id]))
        return result
    def search_request(self,Socket,client):
        print("{} is connected".format(client))
        data = Socket.recv(1024)
        while data:
            tmp = json.loads(data)
            print('searching {}...'.format(tmp))
            result = self.search(tmp)
            Socket.send(bytes(json.dumps(self.search(tmp)).encode('utf-8')))
            data = Socket.recv(1024)
        Socket.close()
        print("{} is disconnected".format(client))
    
    def run(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(self.address)
        server.listen(10)
        while True:
            Socket,client = server.accept()
            p = Thread(target = self.search_request,args = (Socket,client))
            p.start()
        """
        TODO：请在服务器端实现合理的并发处理方案，使得服务器端能够处理多个客户端发来的请求
        """
    
        """
        TODO: 请补充实现文本检索，以及服务器端与客户端之间的通信
        
        1. 接受客户端传递的数据， 例如检索词
        2. 调用检索函数，根据检索词完成检索
        3. 将检索结果发送给客户端，具体的数据格式可以自己定义
        
        """

server = LocalServer("127.0.0.1", 1234)
server.run()