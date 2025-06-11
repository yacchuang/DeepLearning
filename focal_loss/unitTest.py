# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:13:58 2025

@author: ya-chen.chuang
"""

import unittest
import numpy as np
from focalLossFunction import BinarySemanticLoss  

class TestBinarySemanticLoss(unittest.TestCase):

    def test_BCE_loss_basic(self):
        """Test BCE loss with simple binary inputs (uint8 and float)"""
        y_pred = np.array([[230, 10], [200, 50]], dtype=np.uint8)
        y_true = np.array([[255, 0], [255, 0]], dtype=np.uint8)

        BinararyLoss = BinarySemanticLoss(classWeight = 0.5, alpha=0.5, gamma=0)
        loss = BinararyLoss.focal_loss(y_pred, y_true)
        self.assertGreater(loss, 0)  # Loss should be positive

    def test_focal_loss_uint8(self):
        """Test focal loss with uint8 inputs"""
        y_pred = np.array([[250, 20], [210, 60]], dtype=np.uint8)
        y_true = np.array([[255, 0], [255, 0]], dtype=np.uint8)

        BinararyLoss = BinarySemanticLoss(classWeight = 1, alpha=0.5, gamma=2)
        focal_loss = BinararyLoss.focal_loss(y_pred, y_true)
        self.assertGreater(focal_loss, 0)  # Loss should be positive

    def test_masking_uint8(self):
        """Test loss computation with a mask applied (uint8 inputs)"""
        y_pred = np.array([[250, 20], [210, 60]], dtype=np.uint8)
        y_true = np.array([[255, 0], [255, 0]], dtype=np.uint8)
        mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)

        BinararyLoss = BinarySemanticLoss(mask=mask)
        loss = BinararyLoss.BCE_loss(y_pred, y_true)
        self.assertGreater(loss, 0)  # Should still be positive, computed on valid pixels

if __name__ == '__main__':
    unittest.main()