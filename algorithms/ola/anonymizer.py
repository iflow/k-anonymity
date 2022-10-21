# -*- coding: utf-8 -*-
"""
main module for OLA
"""

import copy
import sys
import os

from .ola_anonymization import \
    OLA_Anonymization
sys.path.insert(1, os.path.join(sys.path[0], '..'))


def ola_anonymize(k, att_trees, data, qi_index, qi_names, sa_index, **kwargs):
    """
     Optimal Lattice Anonymization Anonymization
    """
    result, runtime = OLA_Anonymization(att_trees, copy.deepcopy(data), k, qi_index, qi_names, sa_index)

    return result, runtime
