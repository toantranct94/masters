import numpy as np

class SVD:
    def __init__(self):
        self.words = []
        pass

    def read_data_from_csv(self, file_path):
        self.words = [w.lower() for w in self.words]
        pass

    def process(self):
        a = np.array([1,1,1,0,1,1,1,0,0,0,1,0,1,0,0,1,0,1,1,1,1,1,1,1,1,0,1,0,2,0,0,1,1])
        a = a.reshape(11,3)
        u, s, vh = np.linalg.svd(a, full_matrices=False)
        print()

    def bow(self, sentence, words):
        # bag of words
        bag = [0]*len(words)  
        for s in sentence:
            for i,w in enumerate(words): 
                if w == s: 
                    bag[i] = 1

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # svd = SVD()
    # svd.process()
    
    pass