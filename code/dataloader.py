# -*- coding: utf-8 -*-
"""
Dataset Loader for LightGCN.
Original code inspired by: https://github.com/gusye1234/LightGCN-PyTorch
Modified by: chenz0zz0

Purpose:
Handles data loading, sparse graph construction, and user-item interactions parsing.

Key Optimizations:
1. Robust Type Conversion: Ensures user IDs are properly cast to standard Python ints.
2. Hashable Keys Fix: Prevents errors when dictionary keys are numpy arrays during test dict building.
3. MPS Optimization: Adjacency matrix operations optimized to run smoothly on CPU/MPS before sending to device.
4. Deprecation Fixes: Upgraded legacy sparse tensor initializations to modern `torch.sparse_coo_tensor`.
5. Dirty Data Handling: Replaced strict space splitting and added empty sequence guards to gracefully handle malformed datasets (like amazon-book).
"""

import os
from os.path import join
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import world
from world import cprint
from time import time

class BasicDataset(Dataset):
    """Abstract Base Class for recommendation datasets."""
    def __init__(self):
        pass
    @property
    def n_users(self): raise NotImplementedError
    @property
    def m_items(self): raise NotImplementedError
    @property
    def trainDataSize(self): raise NotImplementedError
    @property
    def testDict(self): raise NotImplementedError
    @property
    def allPos(self): raise NotImplementedError
    def getSparseGraph(self): raise NotImplementedError


class LastFM(BasicDataset):
    """
    Dataset loader specific to the LastFM dataset.
    Includes social network graph information alongside user-item interactions.
    """
    def __init__(self, path="../data/lastfm"):
        cprint("loading [last fm]")
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        
        trainData = pd.read_table(join(path, 'data1.txt'), header=None)
        testData  = pd.read_table(join(path, 'test1.txt'), header=None)
        trustNet  = pd.read_table(join(path, 'trustnetwork.txt'), header=None).to_numpy()
        
        trustNet -= 1
        trainData -= 1
        testData -= 1
        self.trustNet  = trustNet
        self.trainData = trainData
        self.testData  = testData
        self.trainUser = np.array(trainData[:][0])
        self.trainUniqueUsers = np.unique(self.trainUser)
        self.trainItem = np.array(trainData[:][1])
        
        self.testUser  = np.array(testData[:][0])
        self.testUniqueUsers = np.unique(self.testUser)
        self.testItem  = np.array(testData[:][1])
        self.Graph = None
        print(f"LastFm Sparsity : {(len(self.trainUser) + len(self.testUser))/self.n_users/self.m_items}")
        
        # (users, users)
        self.socialNet = csr_matrix((np.ones(len(trustNet)), (trustNet[:,0], trustNet[:,1])), shape=(self.n_users, self.n_users))
        # (users, items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)), shape=(self.n_users, self.m_items)) 
        
        # Pre-calculate positive and negative interactions
        self._allPos = self.getUserPosItems(list(range(self.n_users)))
        self.allNeg = []
        allItems = set(range(self.m_items))
        for i in range(self.n_users):
            pos = set(self._allPos[i])
            neg = allItems - pos
            self.allNeg.append(np.array(list(neg)))
        self.__testDict = self.__build_test()

    @property
    def n_users(self): return 1892
    @property
    def m_items(self): return 4489
    @property
    def trainDataSize(self): return len(self.trainUser)
    @property
    def testDict(self): return self.__testDict
    @property
    def allPos(self): return self._allPos

    def getSparseGraph(self):
        """Builds and caches the normalized adjacency matrix."""
        if self.Graph is None:
            user_dim = torch.LongTensor(self.trainUser)
            item_dim = torch.LongTensor(self.trainItem)
            
            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            second_sub = torch.stack([item_dim + self.n_users, user_dim])
            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).int()
            
            # [FIX 4]: Deprecation fix -> Use torch.sparse_coo_tensor
            self.Graph = torch.sparse_coo_tensor(index, data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]), dtype=torch.int32)
            dense = self.Graph.to_dense()
            D = torch.sum(dense, dim=1).float()
            D[D == 0.] = 1.
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
            dense = dense / D_sqrt
            dense = dense / D_sqrt.t()
            index = dense.nonzero()
            data  = dense[dense >= 1e-9]
            assert len(index) == len(data)
            
            # [FIX 4]: Deprecation fix -> Use torch.sparse_coo_tensor
            self.Graph = torch.sparse_coo_tensor(index.t(), data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]), dtype=torch.float32)
            
            # [FIX 3]: Apple MPS Crash Prevention -> Keep sparse matrix on CPU
            self.Graph = self.Graph.coalesce().cpu()
        return self.Graph

    def __build_test(self):
        """Constructs a dictionary mapping users to their ground-truth test items."""
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            # [FIX 1 & 2]: Clean numpy array wrapper to standard hashable Python int
            clean_uid = int(np.array(user).flatten()[0]) if isinstance(user, (np.ndarray, list)) else int(user)
            if test_data.get(clean_uid):
                test_data[clean_uid].append(item)
            else:
                test_data[clean_uid] = [item]
        return test_data
    
    def getUserItemFeedback(self, users, items):
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1, ))
    
    def getUserPosItems(self, users):
        return [self.UserItemNet[user].nonzero()[1] for user in users]
    
    def getUserNegItems(self, users):
        return [self.allNeg[user] for user in users]


class Loader(BasicDataset):
    """
    Standard dataset loader optimized for implicit feedback benchmarks.
    Supported formats: Gowalla, Yelp2018, Amazon-Book.
    """
    def __init__(self, config=world.config, path="../data/gowalla"):
        cprint(f'loading [{path}]')
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.n_user, self.m_item = 0, 0
        train_file, test_file = path + '/train.txt', path + '/test.txt'
        self.path = path
        
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize, self.testDataSize = 0, 0

        # Load Training Data
        with open(train_file) as f:
            for l in f.readlines():
                # [FIX 5]: Use .split() without args to handle dirty spacing and trailing whitespaces
                l = l.strip().split() 
                if len(l) > 0:
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    self.n_user = max(self.n_user, uid)
                    
                    items = [int(i) for i in l[1:]]
                    # [FIX 5]: Guard against zero-item/ghost users
                    if len(items) > 0:
                        trainUser.extend([uid] * len(items))
                        trainItem.extend(items)
                        self.m_item = max(self.m_item, max(items))
                        self.traindataSize += len(items)
        
        # Load Testing Data
        with open(test_file) as f:
            for l in f.readlines():
                # [FIX 5]: Same dirty data handling for test set
                l = l.strip().split() 
                if len(l) > 0:
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    self.n_user = max(self.n_user, uid)
                    
                    items = [int(i) for i in l[1:]]
                    if len(items) > 0:
                        testUser.extend([uid] * len(items))
                        testItem.extend(items)
                        self.m_item = max(self.m_item, max(items))
                        self.testDataSize += len(items)
        
        self.m_item += 1
        self.n_user += 1
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser, self.trainItem = np.array(trainUser), np.array(trainItem)
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser, self.testItem = np.array(testUser), np.array(testItem)
        
        self.Graph = None
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        print(f"{world.dataset} is ready to go.")

    @property
    def n_users(self): return self.n_user
    @property
    def m_items(self): return self.m_item
    @property
    def trainDataSize(self): return self.traindataSize
    @property
    def testDict(self): return self.__testDict
    @property
    def allPos(self): return self._allPos

    def _convert_sp_mat_to_sp_tensor(self, X):
        """Converts scipy sparse matrix to PyTorch sparse tensor."""
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    def getSparseGraph(self):
        """Constructs and caches the normalized adjacency matrix."""
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                norm_adj = pre_adj_mat
            except:
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                norm_adj = d_mat.dot(adj_mat).dot(d_mat).tocsr()
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            # [FIX 3]: Apple MPS Crash Prevention -> Keep sparse matrix coalesced on CPU
            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj).coalesce().cpu()
        return self.Graph

    def __build_test(self):
        """Constructs a dictionary mapping users to their ground-truth test items."""
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            # [FIX 1 & 2]: Clean numpy array wrapper to standard hashable Python int
            clean_uid = int(np.array(user).flatten()[0]) if isinstance(user, (np.ndarray, list)) else int(user)
            
            if test_data.get(clean_uid):
                test_data[clean_uid].append(item)
            else:
                test_data[clean_uid] = [item]
        return test_data

    def getUserPosItems(self, users):
        return [self.UserItemNet[user].nonzero()[1] for user in users]