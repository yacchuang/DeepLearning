# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:13:58 2025

@author: ya-chen.chuang
"""
import numpy as np


class BinarySemanticLoss:
    
    def __init__(self, 
                 classWeight = 1,  
                 alpha=1,
                 gamma=2,
                 mask=None):
        
        """
        alpha: weight factor for class balancing in focal loss
        gamma: if gamma = 0, focal loss is equivalent to BCE loss
        classWeight: default to 1 for applying focal loss, >1 for more influence on positive pixels
        mask: mask (ndarray, optional): A binary array of shape (Height, Width), where 1 indicates valid pixels and 0 indicates bad pixels to be ignored from the loss.
        """
        
        super().__init__()
        self.classWeight = classWeight
        self.alpha = alpha
        self.gamma = gamma
        self.mask = mask

    
    
    def BCE_loss(self, y_pred, y_true):
          
        y_pred = y_pred.astype(np.float32) / 255. if y_pred.dtype == np.uint8 else y_pred
        y_true = y_true.astype(np.float32) / 255. if y_true.dtype == np.uint8 else y_true
        
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)   
        loss = - (self.classWeight * y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))  # alpha already implements class weight, if focal loss is applied, self.classWeight can be set to 1
        
        if self.mask is not None:
            loss = loss * self.mask  # Zero out ignored pixels
            BCELoss = np.sum(loss) / np.sum(self.mask)  # Normalize over valid pixels
        else:
            BCELoss = np.mean(loss)
        
        return BCELoss
    

    def focal_loss(self, y_pred, y_true):
    
        bce_loss= self.BCE_loss(y_pred, y_true)        
        focal_loss = self.alpha * (1 - np.exp(-bce_loss))** self.gamma * bce_loss 
        
        return focal_loss
        

if __name__ == '__main__': 
    
    # test 1: random
    y_true1 = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ])

    y_pred1 = np.array([
        [0.9, 0.8, 0.9],  
        [0.7, 0.2, 0.8],  
        [0.9, 0.8, 0.9] 
    ])
    
    # test 2: good prediction
    y_true2 = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])

    y_pred2 = np.array([
        [0.95, 0.92, 0.94],
        [0.93, 0.08, 0.91], 
        [0.96, 0.94, 0.95]
    ])
    
    # test 3: bad prediction
    y_true3 = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])
    
    y_pred3 = np.array([
        [0.15, 0.25, 0.30],  
        [0.20, 0.85, 0.10],  
        [0.25, 0.15, 0.20]   
    ])
    
    BinararyLoss = BinarySemanticLoss(classWeight = 0.5, alpha=1, gamma=2, mask=None)    
    loss = BinararyLoss.focal_loss(y_pred1, y_true1)