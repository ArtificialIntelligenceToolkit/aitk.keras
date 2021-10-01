# -*- coding: utf-8 -*-
# **************************************************************
# aitk.keras: A Python Keras model API
#
# Copyright (c) 2021 AITK Developers
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.keras
#
# **************************************************************

import numpy as np

class ToleranceAccuracy():
    def __init__(self, tolerance):
        self.tolerance = tolerance
        self.name = "tolerance_accuracy"
        self.reset_state()

    def reset_state(self):
        self.accurate = 0
        self.total = 0

    def update_state(self, targets, outputs):
        results = np.all(
            np.less_equal(np.abs(targets - outputs),
                          self.tolerance), axis=-1)
        self.accurate += sum(results)
        self.total += len(results)

    def result(self):
        return self.accurate / self.total
