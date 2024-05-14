import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Plt(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def fig_size(self,x_size,y_size):
        self.x_size = x_size
        self.y_size = y_size 
        return self
    def plt_show(self,label = None ):
        plt.figure(figsize=(self.x_size, self.y_size))
        if label is not None:
            plt.title('y = '+ label)
        plt.plot(self.x, self.y)
        plt.show()



class Sampler(object):
    def __init__(self,data,batch_size = 10):
        self.data = data
        self.row, self.column = data.shape[0],data.shape[1]
        self.dtype = data.dtype
        self.batch_size = batch_size
        
    def transform_data(self):
        quotient, remainder = divmod(self.row, self.batch_size)
        if remainder != 0:
            return quotient+1,np.concatenate((self.data,np.zeros((self.batch_size-remainder,self.column),dtype = self.dtype )),axis=0)
        else:
            return quotient,self.data
        
    
    def gene_batch(self,T_seq = 10):
        quotient,data = self.transform_data()
        row,column = data.shape[0],data.shape[1]
        gene_index = np.empty(row,dtype=int) 
        gene_index_matrix = np.arange(row).reshape((self.batch_size,quotient))
        sum_index= 0
        for i in range(0,quotient,T_seq):
            batch_index_matrix = gene_index_matrix[:,i:i+T_seq]
            batch_index_matrix_list = batch_index_matrix.flatten()
            gene_index[sum_index:sum_index+batch_index_matrix_list.shape[0]]=batch_index_matrix_list
            sum_index += batch_index_matrix_list.shape[0]
        if sum_index!=row:
            raise RuntimeError("Generate new index error! Need Check!")
        batch_data = np.take(data,gene_index,axis=0)
        return batch_data

class MyDataIterator(object):
    def __init__(self,features,labels,batch_size,T_seq):
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.T_seq = T_seq
        self.num_batches,  remainder = divmod(features.shape[0],batch_size*T_seq)
        if remainder !=0:
            self.num_batches+=1

    def __iter__(self):
        self.iter_indices = 0
        return self

    def __next__(self):
        if self.iter_indices >= self.num_batches:
            raise StopIteration
        start_idx = self.iter_indices * self.batch_size*self.T_seq
        end_idx = min((self.iter_indices + 1) * self.batch_size*self.T_seq, self.features.shape[0])
        self.iter_indices += 1
        return self.transform_batch_data(start_idx,end_idx)
        
    def transform_batch_data(self,start_idx,end_idx):
        batch_features,batch_labels = self.features[start_idx:end_idx,:],self.labels[start_idx:end_idx,:]
        return batch_features.reshape((self.batch_size,-1,self.features.shape[1])),batch_labels.reshape((self.batch_size,-1,self.labels.shape[1]))
        


