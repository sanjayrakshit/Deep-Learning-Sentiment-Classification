import os
import re
import string 
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
import glob
from tqdm import tqdm
import numpy as np
import random



class Load_data:
    def __init__(self, batch_size, sequence_length):
        self.batch_size = batch_size
        self.train_path = os.path.join(".", "aclImdb", "train")
        self.test_path = os.path.join(".", "aclImdb", "test")
        self.sequence_length = sequence_length


    def preprocess(self, x):
        x = re.sub('<[^>]*>', '', x.lower())
        for punc in string.punctuation:
            if "\'" != punc:
                x = x.replace(punc, f" {punc} ")
        x = re.sub(" +", " ", x)
        return x


    def create_ids(self):
        print("Creating ids...")
        all_reviews = ""
        for folder in [os.path.join(self.train_path, "pos"), os.path.join(self.train_path, "neg")]:
            for file in tqdm(glob.glob(os.path.join(folder, "*.txt"))):
                all_reviews += self.preprocess(open(file).read().strip()) + " "
        tokens = dict(Counter(all_reviews.strip().split())).keys()
        self.wids = {
            item: index+1 for index, item in enumerate(tokens)
        }

    
    def parse_file(self):
        print("Parsing train data...")
        train_x, train_y, test_x, test_y = [], [], [], []
        for folder in [os.path.join(self.train_path, "pos"), os.path.join(self.train_path, "neg")]:
            for file in tqdm(glob.glob(os.path.join(folder, "*.txt"))):
                review = self.preprocess(open(file).read().strip())
                review = [self.wids.get(item, len(self.wids)+1) for item in review.split()]
                train_x.append(review)
                if folder.endswith("pos"):
                    train_y.append(np.array([0, 1]))
                else:
                    train_y.append(np.array([1, 0]))
        
        print("Parsing test data...")
        rows = []
        for folder in [os.path.join(self.test_path, "pos"), os.path.join(self.test_path, "neg")]:
            for file in tqdm(glob.glob(os.path.join(folder, "*.txt"))):
                review = self.preprocess(open(file).read().strip())
                review = [self.wids.get(item, len(self.wids)+1) for item in review.split()]
                test_x.append(review)
                if folder.endswith("pos"):
                    test_y.append(np.array([0, 1]))
                else:
                    test_y.append(np.array([1, 0]))
        
        train_x = pad_sequences(train_x, maxlen=self.sequence_length)
        test_x = pad_sequences(test_x, maxlen=self.sequence_length)
        self.train = random.shuffle(list(zip(train_x, train_y)))
        self.test = random.shuffle(list(zip(test_x, test_y)))


    def get_train_batch(self, i):
        temp = list(zip(*self.train[i*self.batch_size : (i+1)*self.batch_size]))
        return temp[0], temp[1]


    def get_test_batch(self, i):
        temp = list(zip(*self.test[i*self.batch_size : (i+1)*self.batch_size]))
        return temp[0], temp[1]


if __name__ == '__main__':
    l = Load_data(batch_size=128, sequence_length=200)       
    l.create_ids()
    l.parse_file()
    batch = l.get_train_batch(i=0)
    print(batch)





















