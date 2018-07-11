# Copyright 2018 Jörg Franke
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import numpy as np
from collections import deque

"""
EarlyStop has a true call return if the loss was the last "list_len" higher as the loss before.
"""

class EarlyStop():
    def __init__(self, list_len=5):

        self.loss_list = deque()
        self.list_len = list_len

    def __call__(self, loss):

        self.loss_list.append(loss)

        if self.loss_list.__len__() > self.list_len:

            self.loss_list.rotate(-1)
            self.loss_list.pop()

            if all(np.around(self.loss_list[i], 3) < np.around(self.loss_list[i + 1], 3) for i in
                   range(len(self.loss_list) - 1)):
                return True
            else:
                return False
        else:
            return False
