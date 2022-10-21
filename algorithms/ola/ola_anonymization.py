# -*- coding: utf-8 -*-
"""
Implements the OLA method
"""

from .lattice import make_lattice, Node
from .information_loss import prec_loss
from .utils import df_to_values, k_anonymity_check
from .generalizations import GenRule, GenMapper

import time
import math
import pandas as pd


def _add_k_minimal(node, k_min_set):
    """ Add node to k-minimal set, removing all higher nodes in path to leaf """
    to_remove = []

    for old_node in k_min_set:
        if node.has_descendant(old_node):
            to_remove.append(old_node)

    for doomed in to_remove:
        k_min_set.remove(doomed)

    k_min_set.add(node)


def _check_kanonymity(df, node, k, max_sup):
    return k_anonymity_check(df, node.gen_rules.keys(), k, max_sup)


def _k_min(b_node, t_node, k, max_sup, k_min_set=set()):
    """ Core of OLA's operation: build k-minimal set with binary search in generalization
    strategies of lattice """
    lattice_lvls = make_lattice(b_node, t_node)
    h = len(lattice_lvls)

    if h > 2:
        # look halfway between top and bottom node
        h = math.floor(h/2)
        for n in lattice_lvls[h]:
            if n.suitable_tag == True:
                _k_min(b_node, n, k, max_sup, k_min_set)
            elif n.suitable_tag == False:
                _k_min(n, t_node, k, max_sup, k_min_set)
            elif n.is_suitable(k, max_sup):
                n.set_suitable()
                _k_min(b_node, n, k, max_sup, k_min_set)
            else:
                n.set_non_suitable()
                _k_min(n, t_node, k, max_sup, k_min_set)

    else: # special case of a 2-node lattice
        if b_node.suitable_tag == False:
            n = t_node
        # It's not possible to know that b_node is k_anonymous. Otherwise it would have been selected as top
        # But it's possible across different strategies! (e.g.: if root is k-anonymous)
        elif b_node.suitable_tag == True:
            n = b_node
        elif b_node.is_suitable(k, max_sup):
            b_node.set_suitable()
            n = b_node
        else:
            b_node.set_non_suitable()
            n = t_node

        if n.suitable_tag == True:
            _add_k_minimal(n, k_min_set)
        elif n.is_suitable(k, max_sup):
            n.set_suitable()
            _add_k_minimal(n, k_min_set)

    return k_min_set


def _make_release(df, qis, k):
    """ Finalize release by suppressing required records and producing some stats """
    records, qi_idx = df_to_values(df, qis)
    original_size = len(records)
    qi_values = lambda record: tuple([record[idx] for idx in qi_idx])

    eq_classes = {}
    release = []

    for r in records:
        qi_signature = qi_values(r)
        if qi_signature in eq_classes.keys():
            eq_classes[qi_signature].append(r)
        else:
            eq_classes[qi_signature] = [r]

    sup_ec = 0
    sup_rec = 0
    for val in eq_classes.values():
        if len(val) >= k:
            release += val
        else:
            sup_ec += 1
            sup_rec += len(val)

    stats = {
        'eq_classes_before_sup': len(eq_classes.keys()),
        'suppressed_classes': sup_ec,
        'suppressed_records': sup_rec,
        'perc_suppressed_records': round((sup_rec/original_size)*100, 2),
    }

    release = pd.DataFrame(release, columns=df.columns)
    return release, stats


def create_data_frame_from_raw_data(data, qi_index, qi_names):
    """
    Creates a dataframe which is compatible to this OLA implementation.
    :param data: raw data
    :param qi_index: quasi identifier indexes
    :param qi_names: quasi identifier names
    :return: dataframe
    """
    df = pd.DataFrame(data)
    df_columns = {}
    for i, name in zip(qi_index, qi_names):
        df_columns[i] = name
    # Set header keys
    df = df.rename(columns=df_columns)
    return df


def tree_to_dict(tree):
    """
    :param tree: list of GenTree objects
    :return: dictionary
    """
    levels = {}
    for name, gen_tree in tree.items():
        if len(gen_tree.leaf_list):
            if gen_tree.level not in levels:
                levels[gen_tree.level] = {}
            for leaf in gen_tree.leaf_list:
                levels[gen_tree.level][str(leaf)] = str(name)
    return levels


def create_generalization_rules(att_trees, qi_names):
    """
    :param att_trees: list of hierarchies
    :param qi_names: quasi identifier names
    :return: generalization rules dictionary
    """
    gen_rules = {}
    for tree, name in zip(att_trees, qi_names):
        rules = []
        hierarchy_dict = tree_to_dict(tree)
        for level, items in hierarchy_dict.items():
            # Do not add the top level ("*"), because it won't be handled by crowd's OLA algorithm.
            if level > 0:
                # Add items at beginning, because we want it ordered
                # from low generalization to high generalization ("*")
                rules.insert(0, GenMapper(items, name, level))
        gen_rules[name] = GenRule(rules)
    return gen_rules


def OLA_Anonymization(att_trees, data, k, qi_index, qi_names, sa_index):
    """
    TODO Description
    """
    max_sup = 0.0
    info_loss = prec_loss # entropy_loss
    df = create_data_frame_from_raw_data(data, qi_index, qi_names)
    generalization_rules = create_generalization_rules(att_trees, qi_names)
    start_time = time.time()

    """ Execute OLA """
    print('Building lattice...')
    b_node, t_node = Node.build_network(generalization_rules, df, _check_kanonymity)

    print('Searching lattice...')
    k_min_nodes = _k_min(b_node, t_node, k, max_sup)

    if len(k_min_nodes) == 0:
        # This cannot happen if, as they should, all generalization rules bring values to indistinguishability
        print('No strategy was found! Aborting.')
        return None

    visited_nodes, checked_nodes, num_suitable, num_non_suitable = b_node.lattice_stats
    print(f"visited {visited_nodes} nodes, checked {checked_nodes} nodes")
    print(f"num k {num_suitable} nodes, num not k {num_non_suitable} nodes")

    print('Choosing optimal generalization strategy...')
    losses = [(info_loss(node), node) for node in k_min_nodes]
    optimal_loss, optimal_node = min(losses, key=lambda x: x[0])

    print(f'Best loss ({optimal_loss}) was obtained with node {optimal_node}')
    release, release_stats = _make_release(optimal_node.apply_gen(), generalization_rules.keys(), k)
    print(release_stats)
    print('Done.')

    rtime = float(time.time() - start_time)

    # To be compatible with the later metric calculations, we must replace NA's
    release = release.fillna(value='*')

    release_list = release.values.tolist()

    return release_list, rtime
