import os
import itertools
import ast
import time
import re
import timeit
import math

from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from py_stringmatching import NeedlemanWunsch, Affine, SmithWaterman, Jaccard, QgramTokenizer
import dask.dataframe as dd
from nltk.stem import WordNetLemmatizer


def block(row, index, n):
    # derived_str = [x for x in row["Derived_Strings"][:n]]
    derived_str = row["Derived_Strings"][:n]
    terms = set()
    # bt_match = set()
    # old solution
    for d in derived_str:
        curr_terms = set()
        # for t in re.split("[ ]", d[0]):
        for t in re.split("[ ]", d):
            if t in index and len(index[t]) < 300:
                curr_terms = curr_terms | index[t]
        terms = terms | curr_terms


    # row['BT_Match'] = bt_match.pop() if len(bt_match) == 1 else None
    # row['Blocking_Candidates'] = [row['BT_Match']] if row['BT_Match'] is not None else list(terms)
    row['Blocking_Candidates'] = list(terms)
    return row


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


def calculate_PR(tab, gold, col_name, bus_term):
    merged_tab = tab.merge(gold, on=[col_name, bus_term], how='outer', indicator=True)
    # merged_tab = tab.merge(gold, on=["col_id", "bt_id"], how='outer', indicator=True)
    FN = len(merged_tab[merged_tab['_merge'] == 'right_only'])
    TP = len(merged_tab[merged_tab['_merge'] == 'both'])
    FP = len(merged_tab[merged_tab['_merge'] == 'left_only'])

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return precision, recall


def create_missing_examples(missing_id, col_table, bt_table, col_name, bus_term):
    # missing_examples = missing_id.merge(col_table, on=["col_id"]).drop(["Derived_Strings"], axis=1)
    missing_examples = missing_id.merge(col_table, on=["col_id"])
    missing_examples = missing_examples.rename({"Tokens": col_name + "_TOKENS"}, axis=1)
    missing_examples = missing_examples.merge(bt_table, on=["bt_id"])
    missing_examples = missing_examples.rename({"Tokens": bus_term + "_TOKENS"}, axis=1)
    return missing_examples


def append_past_train_examples(blocking_cand, train_labels, col_table, bt_table, col_name, bus_term):
    missing_id = blocking_cand.merge(train_labels, on=["col_id", "bt_id"], how='right', indicator=True)
    missing_id = missing_id[missing_id["_merge"] == "right_only"]
    missing_id = missing_id[["col_id", "bt_id"]]
    missing_examples = create_missing_examples(missing_id, col_table, bt_table, col_name, bus_term)
    blocking_cand = blocking_cand.append(missing_examples)
    return blocking_cand.reset_index(drop=True)


# new version
def find_all_derived_matches(col_tok_list, term_tok_list):
    if len(col_tok_list) == 0 or len(term_tok_list) == 0:
        return 0

    derived_matches = []
    for term_tok in term_tok_list:
        curr_matches = []
        for j, col_tok in enumerate(col_tok_list):
            if col_tok[0] == term_tok[0] and \
                    (is_subseq(col_tok, term_tok) or is_subseq(term_tok, col_tok)):
                curr_matches.append(j)

        if len(curr_matches) == 0:
            continue

        derived_matches.append(curr_matches)

    return derived_matches


# new version; should be optimal
def find_highest_derived_overlap(derived_matches):
    sorted_matches = sorted(derived_matches, key=len)

    overlaps = set()
    for d_matches in sorted_matches:
        if len(overlaps) == 0:
            overlaps.add(d_matches[0])
            continue

        for d_match in d_matches:
            if d_match not in overlaps:
                overlaps.add(d_match)
                break

    return len(overlaps)


def remove_stopwords(tok_list):
    return [token for token in tok_list if token not in stop_words]


def remove_starting_trailing_letters(tok_list):
    if len(tok_list) < 2:
        return tok_list

    if len(tok_list[0]) == 1 and tok_list[0].isalpha():
        tok_list = tok_list[1:]

    return tok_list


def preprocess_token_list(tok_list):
    preprocessed_tok_list = remove_stopwords(tok_list)
    preprocessed_tok_list = remove_starting_trailing_letters(preprocessed_tok_list) # remove the first token if the first token is a single letter
    return preprocessed_tok_list


def get_user_pairs(traits, thresh, score_eval, schema):
    # temporary solution that uses precalculated scores
    output = traits[(traits["Precalculated_Score"] >= thresh) & (traits["_merge"] == score_eval)]
    return output[["col_id", schema["col_name"], schema["col_name_tok"], "Derived_Strings", "bt_id",
                   schema["bus_term"], schema["bus_term_tok"]]].sample(2)


def calculate_overlap_coefficient(tok_list1, tok_list2):
    tok_list1 = preprocess_token_list(tok_list1)
    tok_list2 = preprocess_token_list(tok_list2)

    derived_matches = find_all_derived_matches(tok_list1, tok_list2)
    # add a corner case when derived_matches == 0, otherwise find_highest_derived_overlap fails
    if derived_matches == 0:
        return 0
    num_overlap = find_highest_derived_overlap(derived_matches)

    overlap_coeff = num_overlap / min(len(tok_list1), len(tok_list2))
    return overlap_coeff


def calculate_jaccard_coefficient(tok_list1, tok_list2):
    tok_list1 = preprocess_token_list(tok_list1)
    tok_list2 = preprocess_token_list(tok_list2)

    derived_matches = find_all_derived_matches(tok_list1, tok_list2)
    # add a corner case when derived_matches == 0, otherwise find_highest_derived_overlap fails
    if derived_matches == 0:
        return 0
    num_overlap = find_highest_derived_overlap(derived_matches)

    jaccard_coeff = num_overlap / (len(tok_list1) + len(tok_list2) - num_overlap)
    return jaccard_coeff


def calculate_absolute_difference_in_number_tokens(tok_list1, tok_list2):
    tok_list1 = preprocess_token_list(tok_list1)
    tok_list2 = preprocess_token_list(tok_list2)
    return abs(len(tok_list1) - len(tok_list2))


def featurize(samples, schema):
    col_name_tok = schema["col_name_tok"]
    bus_term_tok = schema["bus_term_tok"]
    samples["Concat_Col"] = samples.apply(lambda row: " ".join(row[col_name_tok]), axis=1)
    samples["Concat_BT"] = samples.apply(lambda row: " ".join(row[bus_term_tok]), axis=1)


    samples["Overlap_Coeff"] = samples.apply(lambda row: calculate_overlap_coefficient(row[col_name_tok],
                                                                                       row[bus_term_tok]), axis=1)
    samples["Jaccard_Coeff"] = samples.apply(lambda row: calculate_jaccard_coefficient(row[col_name_tok],
                                                                                       row[bus_term_tok]), axis=1)

    samples["Abbr_Jaccard_Coeff"] = \
        samples.apply(lambda row: calculate_relative_difference_in_number_tokens(row[col_name_tok], row[bus_term_tok]),
                      axis=1)
    samples["Abbr_Jaccard_Coeff_No_Stopwords"] = \
        samples.apply(lambda row: calculate_relative_difference_in_number_tokens(remove_stopwords(row[col_name_tok]),
                                                                                 remove_stopwords(row[bus_term_tok])),
                      axis=1)

    samples["Rel_Diff_Num_Tokens"] = \
        samples.apply(
            lambda row: min((row["Abbr_Jaccard_Coeff"][1] - row["Abbr_Jaccard_Coeff"][0]
                             if row["Abbr_Jaccard_Coeff"][1] != 0.0
                             else len(row[col_name_tok]) + len(row[bus_term_tok])),
                            (row["Abbr_Jaccard_Coeff_No_Stopwords"][1] - row["Abbr_Jaccard_Coeff_No_Stopwords"][0]
                             if row["Abbr_Jaccard_Coeff_No_Stopwords"][1] != 0.0
                             else len(remove_stopwords(row[col_name_tok])) + len(remove_stopwords(row[bus_term_tok])))),
            axis=1)

    samples["Abbr_Jaccard_Coeff"] = \
        samples.apply(lambda row: row["Abbr_Jaccard_Coeff"][0] / row["Abbr_Jaccard_Coeff"][1]
        if row["Abbr_Jaccard_Coeff"][1] != 0.0 else 0.0, axis=1)
    samples["Abbr_Jaccard_Coeff_No_Stopwords"] = \
        samples.apply(lambda row:
                      row["Abbr_Jaccard_Coeff_No_Stopwords"][0] / row["Abbr_Jaccard_Coeff_No_Stopwords"][1]
                      if row["Abbr_Jaccard_Coeff_No_Stopwords"][1] != 0.0 else 0.0, axis=1)

    samples["Affine_Dist"] = samples.apply(lambda row: calculate_affine(row["Concat_Col"], row["Concat_BT"],
                                                                        gap_start=0.8, gap_continuation=0.4,
                                                                        match=2.0, non_match=-1.0), axis=1)
    samples["SmithWaterman_Dist"] = samples.apply(lambda row: calculate_smith_waterman(row["Concat_Col"],
                                                                                       row["Concat_BT"],
                                                                                       gap_cost=0.8, match=2.0,
                                                                                       non_match=-1.0), axis=1)
    samples["SmithWaterman_Coeff"] = samples.apply(lambda row: calculate_smith_waterman_coeff(row["Concat_Col"],
                                                                                              row["Concat_BT"],
                                                                                              gap_cost=0.8, match=2.0,
                                                                                              non_match=-1.0), axis=1)  # TO DO: Haven't been fixed with the new function

    samples["Abs_Diff_Num_Tokens"] = \
        samples.apply(lambda row: calculate_absolute_difference_in_number_tokens(row[col_name_tok],
                                                                                 row[bus_term_tok]), axis=1)

    samples["Normalized_3gram_Jaccard_Coeff"] = samples.apply(lambda row:
                                                              calculate_normalized_3gram_jaccard_coefficient(
                                                                  row["Derived_Strings"], row["Concat_BT"]), axis=1)
    samples["Normalized_SmithWaterman_Dist"] = samples.apply(lambda row:
                                                             calculate_normalized_smith_waterman(
                                                                 row["Derived_Strings"], row["Concat_BT"],
                                                                 gap_cost=1.0, match=2.0, non_match=-1.0),
                                                             axis=1)  # TO DO: Haven't been fixed with the new function
    samples["Normalized_SmithWaterman_Coeff"] = samples.apply(lambda row:
                                                              calculate_normalized_smith_waterman_coeff(
                                                                  row["Derived_Strings"], row["Concat_BT"],
                                                                  gap_cost=1.0, match=2.0, non_match=-1.0), axis=1)  # TO DO: Haven't been fixed with the new function

    return samples[["Overlap_Coeff", "Jaccard_Coeff", "Abbr_Jaccard_Coeff", "Abbr_Jaccard_Coeff_No_Stopwords",
                    "Affine_Dist", "SmithWaterman_Dist", "SmithWaterman_Coeff", "Abs_Diff_Num_Tokens",
                    "Rel_Diff_Num_Tokens", "Normalized_3gram_Jaccard_Coeff", "Normalized_SmithWaterman_Dist",
                    "Normalized_SmithWaterman_Coeff"]]


def featurize_derived_strings(samples, features, schema):
    col_name_tok = schema["col_name_tok"]
    bus_term_tok = schema["bus_term_tok"]
    samples["Concat_Col"] = samples.apply(lambda row: " ".join(row[col_name_tok]), axis=1)
    samples["Concat_BT"] = samples.apply(lambda row: " ".join(row[bus_term_tok]), axis=1)

    samples["Normalized_3gram_Jaccard_Coeff"] = features["Normalized_3gram_Jaccard_Coeff"]
    samples["Normalized_SmithWaterman_Dist"] = features["Normalized_SmithWaterman_Dist"]
    samples["Normalized_SmithWaterman_Coeff"] = features["Normalized_SmithWaterman_Coeff"]
    features = features.drop(["Normalized_3gram_Jaccard_Coeff", "Normalized_SmithWaterman_Dist",
                              "Normalized_SmithWaterman_Coeff"], axis=1)

    features["Normalized_3gram_Jaccard_Coeff"] = samples.apply(lambda row:
                                                               calculate_normalized_3gram_jaccard_coefficient(
                                                                   row["Derived_Strings"], row["Concat_BT"])
                                                               if row["Updated"]
                                                               else row["Normalized_3gram_Jaccard_Coeff"], axis=1)
    features["Normalized_SmithWaterman_Dist"] = samples.apply(lambda row:
                                                              calculate_normalized_smith_waterman(
                                                                  row["Derived_Strings"], row["Concat_BT"],
                                                                  gap_cost=1.0, match=2.0, non_match=-1.0)
                                                              if row["Updated"]
                                                              else row["Normalized_SmithWaterman_Dist"],
                                                              axis=1)  
    features["Normalized_SmithWaterman_Coeff"] = samples.apply(lambda row:
                                                               calculate_normalized_smith_waterman_coeff(
                                                                   row["Derived_Strings"], row["Concat_BT"],
                                                                   gap_cost=1.0, match=2.0, non_match=-1.0)
                                                               if row["Updated"]
                                                               else row["Normalized_SmithWaterman_Coeff"], axis=1)  

    return features[["Overlap_Coeff", "Jaccard_Coeff", "Abbr_Jaccard_Coeff", "Abbr_Jaccard_Coeff_No_Stopwords",
                     "Affine_Dist", "SmithWaterman_Dist", "SmithWaterman_Coeff", "Abs_Diff_Num_Tokens",
                     "Rel_Diff_Num_Tokens", "Normalized_3gram_Jaccard_Coeff", "Normalized_SmithWaterman_Dist",
                     "Normalized_SmithWaterman_Coeff"]]


def featurize_row(row, schema, n):
    col_name_tok = row[schema["col_name_tok"]]
    bus_term_tok = row[schema["bus_term_tok"]]
    concat_col = " ".join(col_name_tok)
    concat_bt = " ".join(bus_term_tok)
    row["Overlap_Coeff"] = calculate_overlap_coefficient(col_name_tok, bus_term_tok)
    row["Jaccard_Coeff"] = calculate_jaccard_coefficient(col_name_tok, bus_term_tok)

    interm_abbr_jacc = calculate_relative_difference_in_number_tokens(col_name_tok, bus_term_tok)
    interm_abbr_jacc_no_sw = calculate_relative_difference_in_number_tokens(remove_stopwords(col_name_tok),
                                                                            remove_stopwords(bus_term_tok))

    row["Rel_Diff_Num_Tokens"] = min((interm_abbr_jacc[1] - interm_abbr_jacc[0] if interm_abbr_jacc[1] != 0.0
                                      else len(col_name_tok) + len(bus_term_tok)),
                                     (interm_abbr_jacc_no_sw[1] - interm_abbr_jacc_no_sw[0]
                                      if interm_abbr_jacc_no_sw[1] != 0.0
                                      else len(remove_stopwords(col_name_tok)) + len(remove_stopwords(bus_term_tok))))

    row["Abbr_Jaccard_Coeff"] = interm_abbr_jacc[0] / interm_abbr_jacc[1] if interm_abbr_jacc[1] != 0.0 else 0.0
    row["Abbr_Jaccard_Coeff_No_Stopwords"] = interm_abbr_jacc_no_sw[0] / interm_abbr_jacc_no_sw[1] \
        if interm_abbr_jacc_no_sw[1] != 0.0 else 0.0

    row["Affine_Dist"] = calculate_affine(concat_col, concat_bt, gap_start=0.8, gap_continuation=0.4,
                                          match=2.0, non_match=-1.0)
    row["SmithWaterman_Dist"] = calculate_smith_waterman(concat_col, concat_bt, gap_cost=0.8, match=2.0, non_match=-1.0)
    row["SmithWaterman_Coeff"] = \
        calculate_smith_waterman_coeff(row["SmithWaterman_Dist"], concat_col, concat_bt, match=2.0)

    row["Abs_Diff_Num_Tokens"] = calculate_absolute_difference_in_number_tokens(col_name_tok, bus_term_tok)

    row["Normalized_3gram_Jaccard_Coeff"] = calculate_normalized_3gram_jaccard_coefficient(row["Derived_Strings"],
                                                                                           concat_bt, n)
    sw_dist, index = calculate_normalized_smith_waterman(row["Derived_Strings"], n, concat_bt, gap_cost=1.0,
                                                         match=2.0, non_match=-1.0)
    row["Normalized_SmithWaterman_Dist"] = sw_dist
    row["Normalized_SmithWaterman_Coeff"] = \
        calculate_normalized_smith_waterman_coeff(sw_dist, index, row["Derived_Strings"], concat_bt, match=2.0)

    return row[["Overlap_Coeff", "Jaccard_Coeff", "Abbr_Jaccard_Coeff", "Abbr_Jaccard_Coeff_No_Stopwords",
                "Affine_Dist", "SmithWaterman_Dist", "SmithWaterman_Coeff", "Abs_Diff_Num_Tokens",
                "Rel_Diff_Num_Tokens", "Normalized_3gram_Jaccard_Coeff", "Normalized_SmithWaterman_Dist",
                "Normalized_SmithWaterman_Coeff"]]


def featurize_row_derived_strings(row, schema, n):
    bus_term_tok = row[schema["bus_term_tok"]]
    concat_bt = " ".join(bus_term_tok)

    if row["Updated"]:
        row["Normalized_3gram_Jaccard_Coeff"] = \
            calculate_normalized_3gram_jaccard_coefficient(row["Derived_Strings"], concat_bt, n)
        sw_dist, index = calculate_normalized_smith_waterman(row["Derived_Strings"], n, concat_bt,
                                                             gap_cost=1.0, match=2.0, non_match=-1.0)
        row["Normalized_SmithWaterman_Dist"] = sw_dist
        row["Normalized_SmithWaterman_Coeff"] = \
            calculate_normalized_smith_waterman_coeff(sw_dist, index, row["Derived_Strings"], concat_bt, match=2.0)

    return row[["Overlap_Coeff", "Jaccard_Coeff", "Abbr_Jaccard_Coeff", "Abbr_Jaccard_Coeff_No_Stopwords",
                "Affine_Dist", "SmithWaterman_Dist", "SmithWaterman_Coeff", "Abs_Diff_Num_Tokens",
                "Rel_Diff_Num_Tokens", "Normalized_3gram_Jaccard_Coeff", "Normalized_SmithWaterman_Dist",
                "Normalized_SmithWaterman_Coeff"]]


def multicore_featurize(samples, schema, can_optimize, n):
    dask_samples = dd.from_pandas(samples, npartitions=48)
    if not can_optimize:
        features = dask_samples.map_partitions((lambda df: df.apply(lambda row: featurize_row(row, schema, n), axis=1)),
                                           meta={"Overlap_Coeff": "f8", "Jaccard_Coeff": "f8",
                                                 "Abbr_Jaccard_Coeff": "f8", "Abbr_Jaccard_Coeff_No_Stopwords": "f8",
                                                 "Affine_Dist": "f8",
                                                 "SmithWaterman_Dist": "f8", "SmithWaterman_Coeff": "f8",
                                                 "Abs_Diff_Num_Tokens": "f8", "Rel_Diff_Num_Tokens": "f8",
                                                 "Normalized_3gram_Jaccard_Coeff": "f8",
                                                 "Normalized_SmithWaterman_Dist": "f8",
                                                 "Normalized_SmithWaterman_Coeff": "f8"}).compute(scheduler="processes")
    else:
        features = dask_samples.map_partitions((lambda df: df.apply(lambda row:
                                                                    featurize_row(row, schema, n)
                                                                    if math.isnan(row["Overlap_Coeff"])
                                                                    else featurize_row_derived_strings(row, schema, n),
                                                                    axis=1)),
                                           meta={"Overlap_Coeff": "f8", "Jaccard_Coeff": "f8",
                                                 "Abbr_Jaccard_Coeff": "f8", "Abbr_Jaccard_Coeff_No_Stopwords": "f8",
                                                 "Affine_Dist": "f8",
                                                 "SmithWaterman_Dist": "f8", "SmithWaterman_Coeff": "f8",
                                                 "Abs_Diff_Num_Tokens": "f8", "Rel_Diff_Num_Tokens": "f8",
                                                 "Normalized_3gram_Jaccard_Coeff": "f8",
                                                 "Normalized_SmithWaterman_Dist": "f8",
                                                 "Normalized_SmithWaterman_Coeff": "f8"}).compute(scheduler="processes")

    return features


def multicore_featurize_derived_strings(samples, features, schema, n):
    modified_features = pd.concat([features, samples[[schema["bus_term_tok"], "Derived_Strings", "Updated"]]], axis=1)
    dask_features = dd.from_pandas(modified_features, npartitions=48)
    return dask_features.map_partitions((lambda df: df.apply(lambda row:
                                                             featurize_row_derived_strings(row, schema, n), axis=1)),
                                        meta={"Overlap_Coeff": "f8", "Jaccard_Coeff": "f8", "Abbr_Jaccard_Coeff": "f8",
                                              "Abbr_Jaccard_Coeff_No_Stopwords": "f8", "Affine_Dist": "f8",
                                              "SmithWaterman_Dist": "f8", "SmithWaterman_Coeff": "f8",
                                              "Abs_Diff_Num_Tokens": "f8", "Rel_Diff_Num_Tokens": "f8",
                                              "Normalized_3gram_Jaccard_Coeff": "f8",
                                              "Normalized_SmithWaterman_Dist": "f8",
                                              "Normalized_SmithWaterman_Coeff": "f8"}).compute(scheduler="processes")


def entropy(p1, p2):
    log_p1 = 0 if p1 == 0 else np.log2(p1)
    log_p2 = 0 if p2 == 0 else np.log2(p2)
    return -1 * (p1 * log_p1 + p2 * log_p2)


def get_labels(traits, indexes):
    output = traits.loc[indexes]
    output["Label"] = output.apply(lambda row: True if row["_merge"] == "both" else False, axis=1)
    return output[["col_id", "bt_id", "Label"]]


def get_positive_seeds(features, traits):
    # output1 = features[(features["Abbr_Jaccard_Coeff_No_Stopwords"] >= 0.83) & (features["SmithWaterman_Dist"] >= 45)]
    output1 = features[(features["Jaccard_Coeff"] >= 0.55) & (features["Abbr_Jaccard_Coeff_No_Stopwords"] >= 0.85) &
                       (features["SmithWaterman_Dist"] >= 39)]
    output2 = features[(features["Jaccard_Coeff"] >= 0.8) & (features["Abbr_Jaccard_Coeff"] >= 0.8) &
                       (features["Abbr_Jaccard_Coeff_No_Stopwords"] >= 0.8) & (features["SmithWaterman_Coeff"] >= 0.76)]
    output3 = features[(features["Jaccard_Coeff"] < features["Normalized_3gram_Jaccard_Coeff"]) &
                       (features["Normalized_SmithWaterman_Coeff"] >= 0.8) &
                       (features["Normalized_3gram_Jaccard_Coeff"] >= 0.83)]
    output4 = features[(features["Rel_Diff_Num_Tokens"] == 0) &
                       ((features["Abbr_Jaccard_Coeff_No_Stopwords"] <= 0.4) | (features["Affine_Dist"] >= 34))]
    output5 = features[(features["Rel_Diff_Num_Tokens"] == 1) & (features["SmithWaterman_Dist"] > 44)]
    output6 = features[(features["Rel_Diff_Num_Tokens"] == 2) & (features["Jaccard_Coeff"] == 1.0)]

    output = output1.append(output2) # change this
    output = output.append(output3)
    output = output.append(output4)
    output = output.append(output5)
    output = output.append(output6)
    train_labels = traits.loc[output.index.values][["col_id", "bt_id"]].drop_duplicates()
    train_labels["Label"] = True
    # traits.loc[output.index.values][traits["_merge"] == "left_only"]
    train_features = features.loc[train_labels.index.values]
    return train_features, train_labels


def get_negative_seeds(features, traits, pos_labels):
    features = features.drop(list(pos_labels.index.values))
    traits = traits.drop(list(pos_labels.index.values))
    # filtered_features = features[features["Jaccard_Coeff"] <= 0.4]
    train_features = features.sample(len(pos_labels))
    train_labels = traits.loc[train_features.index.values][["col_id", "bt_id"]]
    train_labels["Label"] = False
    return train_features, train_labels


def train_classifier(tab_name, traits, schema, features, train_labels, num_iteration, curr_epoch, n):
    traits_copy = traits.copy()
    print("Number of updated derived strings in blocking candidates: " + str(len(traits_copy[traits_copy["Updated"]])))
    if curr_epoch < 2:
        can_optimize = False if curr_epoch == 0 else True
        features = multicore_featurize(traits_copy, schema, can_optimize, n)
    else:
        features = multicore_featurize_derived_strings(traits_copy, features, schema, n)
    print("finish featurize")
    features_copy = features.copy()

    train_labels_was_none = False

    if train_labels is None:  # first iteration
        if tab_name != 'NYC':
            pos_train_features, pos_train_labels = get_positive_seeds(features_copy, traits_copy)
            neg_train_features, neg_train_labels = get_negative_seeds(features_copy, traits_copy, pos_train_labels)
        else:
            pos_train_labels = traits_copy.loc[traits_copy[traits_copy['_merge'] == 'both'].sample(n=5).index.values][['col_id', 'bt_id']].drop_duplicates()
            pos_train_labels["Label"] = True
            pos_train_features = features_copy.loc[pos_train_labels.index.values]

            neg_train_labels = traits_copy.loc[traits_copy[traits_copy['_merge'] == 'left_only'].sample(n=5).index.values][['col_id', 'bt_id']].drop_duplicates()
            neg_train_labels["Label"] = False
            neg_train_features = features_copy.loc[neg_train_labels.index.values]

        train_features = pd.concat([pos_train_features, neg_train_features])
        train_labels = pd.concat([pos_train_labels, neg_train_labels])

        train_labels_was_none = True

    else:  # second+ iterations
        traits_copy2 = traits.copy()
        traits_copy2.reset_index(inplace=True)
        train_labels = train_labels.merge(traits_copy2, on=["col_id", "bt_id"])
        train_labels.index = train_labels["index"]
        train_labels = train_labels[["col_id", "bt_id", "Label"]]
        train_features = features_copy.loc[list(train_labels.index.values)]

    features_copy.drop(list(train_features.index.values), inplace=True)

    classifier = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2,
                                        min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                        max_leaf_nodes=None, min_impurity_decrease=0.0,
                                        bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0,
                                        warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
    
    for i in range(0, num_iteration):
        classifier.fit(train_features.values, train_labels["Label"].values)

        probabilities = pd.DataFrame(classifier.predict_proba(features_copy.values), index=features_copy.index)

        # using weighted sampling
        probabilities["Entropy"] = probabilities.apply(lambda row: entropy(row[0], row[1]), axis=1)
        entropies = probabilities[["Entropy"]]
        entropies.sort_values(by="Entropy", ascending=False, inplace=True)

        top_entropies = entropies.head(100)
        try:
            weighted_sample_indexes = top_entropies.sample(10, weights="Entropy").index.values
        except ValueError:
            weighted_sample_indexes = top_entropies.head(10).index.values
        train_features = train_features.append(features_copy.loc[weighted_sample_indexes])
        train_labels = train_labels.append(get_labels(traits, weighted_sample_indexes))
        features_copy.drop(weighted_sample_indexes, inplace=True)

        if len(features_copy) < 100:
            break
    classifier.fit(train_features.values, train_labels["Label"].values)

    return classifier, features, train_labels


def replace_with_user_labels(row):
    if row["_merge"] == "both":
        row["Match"] = row["Label"]
    return row


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


def calculate_abbr_jaccard_greedy_coefficient(list1, list2):
    abbr_matches = generate_abbr_matches(list1, list2)
    matches_perm = list(itertools.product(*abbr_matches)) if len(abbr_matches) != 0 else []

    best_score = 0.0
    ## interval scheduling algorithm
    for perm in matches_perm:
        len_corr1, num_overlaps1, _ = find_best_abbr_overlaps_greedy(perm)
        curr_score1 = (num_overlaps1 + -1*len_corr1) / ((len(list1) + -1*len_corr1) + len(list2) - (num_overlaps1 + -1*len_corr1))
        len_corr2, num_overlaps2, _ = find_best_abbr_overlaps(perm)
        curr_score2 = (num_overlaps2 + -1*len_corr2) / ((len(list1) + -1*len_corr2) + len(list2) - (num_overlaps2 + -1*len_corr2))
        curr_score = max(curr_score1, curr_score2)
        if curr_score > best_score:
            best_score = curr_score

    return best_score


def calculate_relative_difference_in_number_tokens(list1, list2):
    abbr_matches = generate_abbr_matches(list1, list2)
    matches_perm = list(itertools.product(*abbr_matches)) if len(abbr_matches) != 0 else []

    best_score = 0.0
    best_numerator = 0.0
    best_denominator = 0.0
    ## interval scheduling algorithm
    for perm in matches_perm:
        len_corr1, num_overlaps1, _ = find_best_abbr_overlaps_greedy(perm)
        numerator1 = num_overlaps1 + -1*len_corr1
        denominator1 = len(list1) + -1*len_corr1 + len(list2) - (num_overlaps1 + -1*len_corr1)
        curr_score1 = numerator1 / denominator1
        len_corr2, num_overlaps2, _ = find_best_abbr_overlaps(perm)
        numerator2 = num_overlaps2 + -1*len_corr2
        denominator2 = len(list1) + -1*len_corr2 + len(list2) - (num_overlaps2 + -1*len_corr2)
        curr_score2 = numerator2 / denominator2

        if curr_score1 >= curr_score2:
            curr_score = curr_score1
            numerator = numerator1
            denominator = denominator1
        else:
            curr_score = curr_score2
            numerator = numerator2
            denominator = denominator2

        if curr_score > best_score:
            best_score = curr_score
            best_numerator = numerator
            best_denominator = denominator

    return tuple([best_numerator, best_denominator])


def generate_abbr_expansion_lists(list1, list2):
    abbr_matches = generate_abbr_matches(list1, list2)
    matches_perm = list(itertools.product(*abbr_matches)) if len(abbr_matches) != 0 else []

    best_abbr_windows = []
    best_score = 0.0
    ## interval scheduling algorithm
    for perm in matches_perm:
        len_corr1, num_overlaps1, abbr_windows1 = find_best_abbr_overlaps_greedy(perm)
        curr_score1 = num_overlaps1 / (len(list1) + len(list2) + len_corr1 - num_overlaps1)  # TO DO: fix for new scoring method
        len_corr2, num_overlaps2, abbr_windows2 = find_best_abbr_overlaps(perm)
        curr_score2 = num_overlaps2 / (len(list1) + len(list2) + len_corr2 - num_overlaps2)

        if curr_score1 >= curr_score2:
            curr_score = curr_score1
            abbr_windows = abbr_windows1
        else:
            curr_score = curr_score2
            abbr_windows = abbr_windows2

        if curr_score > best_score:
            best_score = curr_score
            best_abbr_windows = abbr_windows

    abbr_expansion_lists = {}
    for window in best_abbr_windows:
        col_token = list1[window[3]]
        bus_token = " ".join(list2[window[0]:window[1]])
        expansions = abbr_expansion_lists[col_token] if col_token in abbr_expansion_lists else set()
        expansions.add(bus_token)
        abbr_expansion_lists[col_token] = expansions

    return abbr_expansion_lists


def calculate_needleman_wunsch(str1, str2):
    nw = NeedlemanWunsch(gap_cost=0.8, sim_func=lambda s1, s2: (2.0 if s1 == s2 else -1.0))
    return nw.get_raw_score(str1, str2)


def calculate_affine(str1, str2, gap_start, gap_continuation, match, non_match):
    aff = Affine(gap_start=gap_start, gap_continuation=gap_continuation,
                 sim_func=lambda s1, s2: (match if s1 == s2 else non_match))
    return aff.get_raw_score(str1, str2)


def calculate_smith_waterman(str1, str2, gap_cost, match, non_match):
    sw = SmithWaterman(gap_cost=gap_cost, sim_func=lambda s1, s2: (match if s1 == s2 else non_match))
    return sw.get_raw_score(str1, str2)


def calculate_normalized_smith_waterman(normalized_tuples, n, bus_term, gap_cost, match, non_match):
    if isinstance(normalized_tuples, float):
        return 0.0, -1

    top_normalized_tuples = normalized_tuples[:n] # hard coded, need to change
    best_score = 0.0
    best_index = -1
    # for i, t in enumerate(top_normalized_tuples):
    #     normalized_col_name = t[0]
    for i, normalized_col_name in enumerate(top_normalized_tuples):
        curr_score = calculate_smith_waterman(normalized_col_name, bus_term, gap_cost=gap_cost,
                                              match=match, non_match=non_match)
        if best_score < curr_score:
            best_score = curr_score
            best_index = i

    return best_score, best_index


def calculate_smith_waterman_coeff(sw_dist, str1, str2, match):
    return sw_dist / (match * max(len(str1), len(str2)))

def calculate_normalized_smith_waterman_coeff(sw_dist, index, normalized_tuples, bus_term, match):
    if sw_dist == 0.0:
        return 0.0

    # return calculate_smith_waterman_coeff(sw_dist, normalized_tuples[index][0], bus_term, match)
    return calculate_smith_waterman_coeff(sw_dist, normalized_tuples[index], bus_term, match)


def calculate_3gram_jaccard_coefficient(str1, str2):
    qg3_tok = QgramTokenizer(qval=3)
    jac = Jaccard()
    return jac.get_raw_score(qg3_tok.tokenize(str1), qg3_tok.tokenize(str2))


def calculate_normalized_3gram_jaccard_coefficient(normalized_tuples, bus_term, n):
    if isinstance(normalized_tuples, float):
        return 0.0

    top_normalized_tuples = normalized_tuples[:n] # HARD-CODED, top r=10 derived column names
    best_score = 0.0
    # for t in top_normalized_tuples:
        # normalized_col_name = t[0]
    for normalized_col_name in top_normalized_tuples:
        best_score = max(best_score, calculate_3gram_jaccard_coefficient(normalized_col_name, bus_term))
    return best_score


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
            curr_score1 = num_overlaps1 / (len(col_list) + len(bt_list) + len_corr1 - num_overlaps1)  # TO DO: fix for new scoring method
            len_corr2, num_overlaps2, abbr_windows2 = find_best_abbr_overlaps(perm)
            curr_score2 = num_overlaps2 / (len(col_list) + len(bt_list) + len_corr2 - num_overlaps2)

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


def update_blocking_cand(blocking_cand, col_table):
    derived_str = col_table[["col_id", "Derived_Strings", "Updated"]]
    new_blocking_cand = blocking_cand.drop(["Derived_Strings", "Updated"], axis=1)
    new_blocking_cand = new_blocking_cand.merge(derived_str, on=["col_id"], how="left")
    return new_blocking_cand


def run_bta_blocking(gold, train_labels, tab_name, col_name, bus_term, n, col_table=None, bt_table=None,
                     inverted_index=None,
                     save=False):
    output_dir = "normalized_output_manualDict_fastText_" + tab_name
    # n = 10
    blocking_cand = col_table.apply(lambda row: block(row, inverted_index, n), axis=1)

    blocking_cand = blocking_cand[
        ["col_id", col_name, "Tokens", "Derived_Strings", "Updated", "Blocking_Candidates"]].explode(
        "Blocking_Candidates")
    # blocking_cand = blocking_cand[["col_id", col_name, "Tokens", "Blocking_Candidates", "BT_Match"]].explode("Blocking_Candidates")
    blocking_cand = blocking_cand.rename({"Tokens": col_name + "_TOKENS", "Blocking_Candidates": bus_term}, axis=1)
    blocking_cand = blocking_cand.merge(bt_table, on=[bus_term])
    blocking_cand = blocking_cand.rename({"Tokens": bus_term + "_TOKENS"}, axis=1)
    if train_labels is not None:
        blocking_cand = append_past_train_examples(blocking_cand, train_labels, col_table, bt_table, col_name, bus_term)

    # blocking_cand["Predicted_Label"] = blocking_cand.apply(lambda row: True if row[bus_term] == row["BT_Match"] else False, axis=1)
    blocking_abs_size = len(blocking_cand)
    cart_prod = len(col_table[[col_name]]) * len(bt_table[[bus_term]])
    blocking_rel_size = blocking_abs_size / cart_prod

    print("Blocking absolute candidate size: " + str(blocking_abs_size))
    print("Blocking relative candidate size: " + str(blocking_rel_size))
    _, blocking_recall = calculate_PR(blocking_cand, gold, col_name, bus_term)
    print("Blocking recall: " + str(blocking_recall))

    # saving blocking outputs
    if save:
        output_filename = tab_name + "_bta_blocking_output_first_iteration_1-70ep_12feat_fixed.csv"
        output_path = os.path.join(output_dir, output_filename)
        blocking_cand.to_csv(output_path, index=False)

    print("done")

    return blocking_cand


def run_bta_matching(blocking_output, gold, features, train_labels, num_iteration, curr_epoch, tab_name, col_name,
                     bus_term, n, save=False):
    output_dir = "normalized_output_manualDict_fastText_" + tab_name
    col_name_tok = col_name + "_TOKENS"
    bus_term_tok = bus_term + "_TOKENS"
    schema = {"col_name": col_name, "col_name_tok": col_name_tok, "bus_term": bus_term, "bus_term_tok": bus_term_tok}

    if curr_epoch < 2:
        # lower-case all tokens
        blocking_output[col_name_tok] = blocking_output[col_name_tok].map(lambda x: [token.lower() for token in x
                                                                                     if len(
                token) > 0 and token.isalnum()])
        blocking_output[bus_term_tok] = blocking_output[bus_term_tok].map(lambda x: [token.lower() for token in x
                                                                                     if len(
                token) > 0 and token.isalnum()])

        print("blocking preprocessing done")

        # blocking_output["Precalculated_Score"] = blocking_output.apply(
        #     lambda row: calculate_jaccard_coefficient(row[col_name_tok], row[bus_term_tok]), axis=1)

    blocking_output = blocking_output.merge(gold, on=[col_name, bus_term], how='left', indicator=True)
    # correct candidate['_merge'] = both, wrong candidate['_merge'] = left_only
    # blocking_output = blocking_output.merge(gold, on=["col_id", "bt_id"], how='left', indicator=True)

    print("successfully loaded temporary user input")

    rf_classifier, featurized_blocking_output, train_labels = train_classifier(tab_name, blocking_output, schema, features,
                                                                               train_labels, num_iteration, curr_epoch, n)
    print("finished training classifier")

    if save:
        with open(os.path.join(output_dir, 'rf_classifier_1-70ep_12feat_fixed.pickle'), 'wb+') as file:
            pickle.dump(rf_classifier, file)

    predictions = pd.DataFrame(rf_classifier.predict(featurized_blocking_output),
                               index=featurized_blocking_output.index,
                               columns=["Match"])

    matches = blocking_output[[col_name, col_name_tok, bus_term, bus_term_tok]]
    matches = matches.merge(featurized_blocking_output, left_index=True, right_index=True)
    matches = matches.merge(predictions, left_index=True, right_index=True)
    matches = matches.merge(train_labels, left_index=True, right_index=True, how="outer", indicator=True)
    if save:  # saving final training labels for debugging
        train_labels_filename = tab_name + "_train_labels_final_iteration_1-70ep_12feat_fixed.csv"
        output_path = os.path.join(output_dir, train_labels_filename)
        merged_train_labels = matches[matches["_merge"] == "both"]
        merged_train_labels = merged_train_labels.drop(["col_id", "bt_id", "_merge"], axis=1)
        merged_train_labels.to_csv(output_path, index=False)
    matches = matches.apply(lambda row: replace_with_user_labels(row), axis=1)
    # matches = matches.drop(["col_id", "bt_id", "Label", "_merge"], axis=1)
    matches = matches.drop(["Label", "_merge"], axis=1)
    print("finished applying classifier")

    matches_only = matches[matches["Match"] == True]
    matching_precision, matching_recall = calculate_PR(matches_only, gold, col_name, bus_term)
    print("Matching precision: " + str(matching_precision*100))
    print("Matching recall: " + str(matching_recall*100))
    print("Matching F1: " + str(2*100*matching_precision*matching_recall/(matching_recall+matching_precision)) )
    print("Total number of TP+FP: " + str(len(matches_only)))
    print("Total number of TP: " + str(matching_precision * len(matches_only)))
    print("Total number of TP+FN: " + str(len(gold)))
    print("Total number of FN: " + str(len(gold) - (matching_precision * len(matches_only))))

    if save:
        output_filename = tab_name + "_bta_output_1-70ep_12feat_fixed.csv"
        output_path = os.path.join(output_dir, output_filename)
        matches.to_csv(output_path, index=False)

    matches_merged = matches.merge(gold, on=[col_name, bus_term], how='outer', indicator=True)
    # matches_merged = matches.merge(gold, on=["col_id", "bt_id"], how='outer', indicator=True)
    conditions = [
        matches_merged["Match"].eq(True) & matches_merged["_merge"].eq("both"),
        matches_merged["Match"].eq(True) & matches_merged["_merge"].eq("left_only"),
        matches_merged["Match"].eq(False) & matches_merged["_merge"].eq("left_only"),
        matches_merged["Match"].eq(False) & matches_merged["_merge"].eq("both"),
        matches_merged["_merge"].eq("right_only")
    ]
    choices = [
        "TP",
        "FP",
        "TN",
        "FN",
        "FN"
    ]
    matches_merged["Score_Eval"] = np.select(conditions, choices, default="other")

    if save:
        output_filename = tab_name + "_bta_output_1-70ep_12feat_fixed_score-eval.csv"
        output_path = os.path.join(output_dir, output_filename)
        matches_merged.to_csv(output_path, index=False)

    print("matching done")

    return matches[[col_name, col_name_tok, bus_term, bus_term_tok, "Match"]], featurized_blocking_output, train_labels


stop_words = set(stopwords.words('english'))
stop_words = stop_words - {"d", "s", "m", "o", "y", "t"}
lem = WordNetLemmatizer()
