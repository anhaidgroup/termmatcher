import os
import itertools
import ast
import time
import re
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from py_stringmatching import NeedlemanWunsch, Affine, SmithWaterman

from bta_cnn_iterator.cnn_initializer import generate_derived_strings


# Taken from https://stackoverflow.com/questions/24017363/how-to-test-if-one-string-is-a-subsequence-of-another
def is_subseq(x, y):
    it = iter(y)
    return all(c in it for c in x)


def is_restricted_subseq(x, y):
    i = 0
    j = 0
    vowels = "aeiou"

    while i < len(x) and j < len(y):
        if x[i] == y[j]:
            if x[i] in vowels and i != 0 and j != 0 and x[i - 1] != y[j - 1]:
                pass
            else:
                i = i + 1
        j = j + 1

    if i == len(x):
        return True
    return False


def find_matching_windows(tokenA, list2, i):
    if i == len(list2):
        return [len(tokenA)]

    first_letter = list2[i][0]
    indices = []
    for j in range(len(tokenA)):
        if j == 0:
            continue
        if tokenA[j] == first_letter:
            # if (j == len(tokenA) - 1) and (len(tokenA) > 2):
            #     continue
            indices.append(j)

    if len(indices) == 0:
        return [len(tokenA)]

    if indices[-1] != len(tokenA):
        indices.append(len(tokenA))

    return indices


def match_abbr(tokenA, list2, i, start):
    if len(tokenA) == 0:  # base case: all letters of abbr have been matched
        return 0
    if i == len(list2):  # base case: not all letters of abbr have been matched, but there are no more tokens left
        return -1 * len(list2)
    if start:
        all_scores = []

    left_idx = 0
    right_idces = find_matching_windows(tokenA, list2, i + 1)
    highest_score = 0
    for right_idx in right_idces:
        curr_abbr = tokenA[left_idx:right_idx]
        if is_subseq(curr_abbr, list2[i]):
            new_score = match_abbr(tokenA[right_idx:], list2, i + 1, False) + 1
            if start and new_score > 0:
                all_scores.append(new_score)
            if curr_abbr == tokenA and curr_abbr == list2[i]:
                highest_score = new_score
                break
            highest_score = max(new_score, highest_score)
        else:
            break

    if highest_score <= 0:
        highest_score = -1 * len(list2)

    if start:
        return all_scores

    return highest_score


def generate_abbr_matches(list1, list2):
    all_abbr_matches = []
    for i, tokenA in enumerate(list1):
        abbr_matches = []
        for j, tokenB in enumerate(list2):
            if tokenA[0] == tokenB[0]:
                num_tokens = match_abbr(tokenA, list2, j, True)
                for num_token in num_tokens:
                    abbr_matches.append((j, j + num_token, num_token, i))

        if len(abbr_matches) != 0:
            all_abbr_matches.append(abbr_matches)
    return all_abbr_matches


def get_non_overlap_window(arr, index):
    for i in range(index - 1, -1, -1):
        if arr[i][1] <= arr[index][0]:
            return i
    return -1


def find_best_abbr_overlaps(windows):
    if len(windows) == 0:
        return 0, 0

    sorted_windows = sorted(windows, key=lambda x: x[1])
    memo_table = [sorted_windows[0][2]]
    non_overlap_window = [-1]
    for i in range(1, len(windows)):
        j = get_non_overlap_window(sorted_windows, i)
        non_overlap_window.append(j)
        if j != -1:
            memo_table.append(max(memo_table[i - 1], sorted_windows[i][2] + memo_table[j]))
        else:
            memo_table.append(max(memo_table[i - 1], sorted_windows[i][2]))

    # backtracking to find actual windows + overlaps
    len_correction = 0
    best_overlaps = 0
    abbr_windows = []
    i = len(windows) - 1
    while i >= 0:
        j = non_overlap_window[i]
        if j == -1:
            memo_j = 0
        else:
            memo_j = memo_table[j]

        if i == 0:
            memo_i_minus_1 = 0
        else:
            memo_i_minus_1 = memo_table[i - 1]

        if sorted_windows[i][2] + memo_j >= memo_i_minus_1:
            len_correction = len_correction + -1 * (sorted_windows[i][2] - 1)
            best_overlaps = best_overlaps + 1
            abbr_windows.append(sorted_windows[j])
            i = j
        else:
            i = i - 1

    return len_correction, best_overlaps, abbr_windows


def find_best_abbr_overlaps_greedy(windows):
    if len(windows) == 0:
        return 0, 0

    sorted_windows = sorted(windows, key=lambda x: x[1])
    i = 0
    len_correction = -1 * (sorted_windows[i][2] - 1)
    best_overlaps = 1
    abbr_windows = [sorted_windows[i]]
    for j in range(1, len(windows)):
        if sorted_windows[j][0] >= sorted_windows[i][1]:
            len_correction = len_correction + -1 * (sorted_windows[j][2] - 1)
            best_overlaps = best_overlaps + 1
            abbr_windows.append(sorted_windows[j])
            i = j

    return len_correction, best_overlaps, abbr_windows


def create_cnn_dict(df):
    local_cnn_dict = {}
    global_cnn_dict = {}
    for row in df.itertuples():
        col_name = row[1]
        col_list = row[2]
        bt_list = row[3]

        abbr_matches = generate_abbr_matches(col_list, bt_list)
        matches_perm = list(itertools.product(*abbr_matches)) if len(abbr_matches) != 0 else []

        best_abbr_windows = []
        best_score = 0.0
        # interval scheduling algorithm
        for perm in matches_perm:
            len_corr1, num_overlaps1, abbr_windows1 = find_best_abbr_overlaps_greedy(perm)
            curr_score1 = (num_overlaps1 + -1 * len_corr1) / (
                    (len(col_list) + -1 * len_corr1) + len(bt_list) - (num_overlaps1 + -1 * len_corr1))
            len_corr2, num_overlaps2, abbr_windows2 = find_best_abbr_overlaps(perm)
            curr_score2 = (num_overlaps2 + -1 * len_corr2) / (
                    (len(col_list) + -1 * len_corr2) + len(bt_list) - (num_overlaps2 + -1 * len_corr2))

            if curr_score1 >= curr_score2:
                curr_score = curr_score1
                abbr_windows = abbr_windows1
            else:
                curr_score = curr_score2
                abbr_windows = abbr_windows2

            if curr_score > best_score:
                best_score = curr_score
                best_abbr_windows = abbr_windows

        abbr_expansion_lists = local_cnn_dict[col_name] if col_name in local_cnn_dict else {}
        for window in best_abbr_windows:
            col_token = col_list[window[3]]
            bus_token = " ".join(bt_list[window[0]:window[1]])

            local_expansions = abbr_expansion_lists[col_token] if col_token in abbr_expansion_lists else set()
            local_expansions.add(bus_token)
            abbr_expansion_lists[col_token] = local_expansions

            global_expansions = global_cnn_dict[col_token] if col_token in global_cnn_dict else set()
            global_expansions.add(bus_token)
            global_cnn_dict[col_token] = global_expansions

        local_cnn_dict[col_name] = abbr_expansion_lists

    return local_cnn_dict, global_cnn_dict


def update_derived_strings(match, col_table, col_name, col_name_tok,
                           bus_term_tok):  # changes col_table; pass by reference
    col = match[col_name]
    col_list = match[col_name_tok]
    bt_list = match[bus_term_tok]

    abbr_matches = generate_abbr_matches(col_list, bt_list)
    matches_perm = list(itertools.product(*abbr_matches)) if len(abbr_matches) != 0 else []

    best_abbr_windows = []
    best_score = 0.0
    # interval scheduling algorithm
    for perm in matches_perm:
        len_corr1, num_overlaps1, abbr_windows1 = find_best_abbr_overlaps_greedy(perm)
        curr_score1 = (num_overlaps1 + -1 * len_corr1) / (
                (len(col_list) + -1 * len_corr1) + len(bt_list) - (num_overlaps1 + -1 * len_corr1))
        len_corr2, num_overlaps2, abbr_windows2 = find_best_abbr_overlaps(perm)
        curr_score2 = (num_overlaps2 + -1 * len_corr2) / (
                (len(col_list) + -1 * len_corr2) + len(bt_list) - (num_overlaps2 + -1 * len_corr2))

        if curr_score1 >= curr_score2:
            curr_score = curr_score1
            abbr_windows = abbr_windows1
        else:
            curr_score = curr_score2
            abbr_windows = abbr_windows2

        if curr_score > best_score:
            best_score = curr_score
            best_abbr_windows = abbr_windows

    if len(best_abbr_windows) != 0:
        abbr_expansion_lists = {}
        for window in best_abbr_windows:
            col_token = col_list[window[3]]
            bus_token = " ".join(bt_list[window[0]:window[1]])

            expansions = abbr_expansion_lists[col_token] if col_token in abbr_expansion_lists else set()
            expansions.add(bus_token)
            abbr_expansion_lists[col_token] = expansions

        old_row = col_table[col_table[col_name] == col]
        derived_strings = set(
            [d for d in old_row["Derived_Strings"].values[0] if d[1] == 1.0])  # not updated for new derived strings
        list_of_lists = []
        for t in old_row["Tokens"].values[0]:
            lowered_t = t.lower()
            if lowered_t in abbr_expansion_lists:
                list_of_lists.append(list(abbr_expansion_lists[lowered_t]))
            else:
                list_of_lists.append([lowered_t])

        derived_tuples = list(itertools.product(*list_of_lists))
        new_derived_strings = [" ".join(t) for t in derived_tuples]

        derived_strings.update([(d, 1.0) for d in new_derived_strings])
        derived_strings = list(derived_strings)
        col_table.at[old_row.index.values[0], "Derived_Strings"] = sorted(derived_strings, key=lambda x: x[1],
                                                                          reverse=True)  

    return


def deduplicate_derived_strings(d_strings):
    d_strings = sorted(d_strings, key=lambda x: (x[0], x[1]), reverse=True)
    i = 0
    dedup_d_strings = []
    while i < len(d_strings):
        dedup_d_strings.append(d_strings[i])
        if i != len(d_strings) - 1 and d_strings[i][0] == d_strings[i + 1][0]:
            i = i + 1
        i = i + 1

    return dedup_d_strings


def update_derived_strings_global(row, col_name, local_cnn_dict, global_cnn_dict):
    col = row[col_name]

    curr_local_cnn_dict = local_cnn_dict[col] if col in local_cnn_dict else None
    list_of_lists = []
    for t in row["Tokens"]:
        lowered_t = t.lower()
        if curr_local_cnn_dict is not None and lowered_t in curr_local_cnn_dict:
            list_of_lists.append(list(curr_local_cnn_dict[lowered_t]))
        elif lowered_t in global_cnn_dict:
            list_of_lists.append(list(global_cnn_dict[lowered_t]))
        else:
            list_of_lists.append([lowered_t])

    derived_tuples = list(itertools.product(*list_of_lists))
    new_derived_strings = [" ".join(t) for t in derived_tuples]

    derived_strings = row["Derived_Strings"]  # not updated for new derived strings

    if curr_local_cnn_dict is not None:
        derived_strings = set([d for d in derived_strings if d[1] == 1.0])
        old_size = len(derived_strings)
        derived_strings.update([(d, 1.0) for d in new_derived_strings])
        new_size = len(derived_strings)
        derived_strings = list(derived_strings)
    else:
        if len(derived_strings) != 0:
            highest_score = derived_strings[0][1]
        else:
            highest_score = 0.5

        derived_strings = set(derived_strings)
        old_size = len(derived_strings)
        derived_strings.update([(d, highest_score) for d in new_derived_strings])
        new_size = len(derived_strings)
        # derived_strings.extend([(d, highest_score) for d in new_derived_strings])
        derived_strings = deduplicate_derived_strings(list(derived_strings))
    derived_strings = sorted(derived_strings, key=lambda x: x[1], reverse=True)

    row["Derived_Strings"] = derived_strings
    row["Updated"] = True if old_size != new_size else False
    return row


def infer_rewrite_rules(df):
    all_rewrite_rules = {}
    for row in df.itertuples():
        col_list = row[2]
        bt_list = row[3]

        abbr_matches = generate_abbr_matches(col_list, bt_list)
        matches_perm = list(itertools.product(*abbr_matches)) if len(abbr_matches) != 0 else []

        best_abbr_windows = []
        best_score = 0.0
        # interval scheduling algorithm
        for perm in matches_perm:
            len_corr1, num_overlaps1, abbr_windows1 = find_best_abbr_overlaps_greedy(perm)
            curr_score1 = (num_overlaps1 + -1 * len_corr1) / (
                    (len(col_list) + -1 * len_corr1) + len(bt_list) - (num_overlaps1 + -1 * len_corr1))
            len_corr2, num_overlaps2, abbr_windows2 = find_best_abbr_overlaps(perm)
            curr_score2 = (num_overlaps2 + -1 * len_corr2) / (
                        (len(col_list) + -1 * len_corr2) + len(bt_list) - (num_overlaps2 + -1 * len_corr2))

            if curr_score1 >= curr_score2:
                curr_score = curr_score1
                abbr_windows = abbr_windows1
            else:
                curr_score = curr_score2
                abbr_windows = abbr_windows2

            if curr_score > best_score:
                best_score = curr_score
                best_abbr_windows = abbr_windows

        # count number of token expansions for each col token
        for window in best_abbr_windows:
            col_token = col_list[window[3]]
            bus_token = " ".join(bt_list[window[0]:window[1]])
            token_rewrite_rules = all_rewrite_rules[col_token] if col_token in all_rewrite_rules else Counter()
            token_rewrite_rules[bus_token] += 1
            all_rewrite_rules[col_token] = token_rewrite_rules

    # score rewrite rules
    for token in all_rewrite_rules:
        token_rewrite_rules = all_rewrite_rules[token]
        count = sum(token_rewrite_rules.values())
        for token_expansion in token_rewrite_rules:
            token_rewrite_rules[token_expansion] = token_rewrite_rules[token_expansion] / count

        scored_token_rewrite_rules = []
        for token_expansion in token_rewrite_rules:
            scored_token_rewrite_rules.append((token_expansion, token_rewrite_rules[token_expansion]))

        all_rewrite_rules[token] = sorted(scored_token_rewrite_rules, key=lambda x: x[1], reverse=True)

    return all_rewrite_rules


def revise_rewrite_rules(row, new_rewrite_rules):
    token_list = row["Tokens"]
    old_rewrite_rules = row["Combined_Candidate_List"]
    has_revised = False

    for i, token in enumerate(token_list):
        lc_token = token.lower()
        if lc_token in new_rewrite_rules:
            if old_rewrite_rules[i] != new_rewrite_rules[lc_token]:
                # we should also just use top_n re_write_rules
                old_rewrite_rules[i] = new_rewrite_rules[lc_token][:3] # hard code to be top 3 rewrite rules for each token
                has_revised = True

    row["Updated"] = False

    if has_revised:
        row["Combined_Candidate_List"] = old_rewrite_rules
        old_derived_strings = row["Derived_Strings"]
        new_derived_strings = generate_derived_strings(old_rewrite_rules)
        if old_derived_strings != new_derived_strings:
            row["Derived_Strings"] = new_derived_strings
            row["Updated"] = True

    return row


def run_cnn(matching_output, col_table, gold, tab_name, col_name, bus_term, save=False):
    col_name_tok = col_name + "_TOKENS"
    bus_term_tok = bus_term + "_TOKENS"
    matches_only = matching_output[~matching_output["Match"].isna()
                                   & matching_output["Match"]][[col_name, col_name_tok, bus_term_tok]]

    rewrite_rules = infer_rewrite_rules(matches_only)
    col_table = col_table.apply(lambda row: revise_rewrite_rules(row, rewrite_rules), axis=1)

    if save:
        output_dir = "normalized_output_manualDict_fastText_" + tab_name
        output_filename = tab_name + "_cnn_output_1-70ep_12feat_fixed.csv"
        output_path = os.path.join(output_dir, output_filename)
        col_table.to_csv(output_path, index=False)
    return col_table


stop_words = set(stopwords.words('english'))
stop_words = stop_words - {"d", "s", "m", "o", "y", "t"}
