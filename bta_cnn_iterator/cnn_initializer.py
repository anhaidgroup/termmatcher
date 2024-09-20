import os
import re
import itertools
import time
from collections import Counter

from sklearn.metrics.pairwise import cosine_similarity
import pickle
import pandas as pd
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.stem import WordNetLemmatizer

from trie import Trie, char_to_index
from bta_cnn_iterator.preprocessor import create_col_table, create_bt_table


def set_type(c):
    if c.isalpha():
        curr_type = "letter"
    elif c.isdigit():
        curr_type = "number"
    else:
        curr_type = "symbol"

    return curr_type


def extract_tokens(row, col_name):
    curr_string = row[col_name]
    curr_token = ''
    curr_type = None
    token_list = []

    for c in curr_string:
        if len(curr_token) == 0:
            curr_token = c
            curr_type = set_type(c)
        else:
            if curr_type is None:
                print("curr_type should not be None!")
            else:
                if c.isalpha():
                    if curr_type == "letter":
                        if c.isupper():
                            if curr_token[-1].isupper():
                                curr_token = ''.join([curr_token, c])
                            else:
                                token_list.append(curr_token)
                                curr_type = set_type(c)
                                curr_token = c
                        else:
                            if len(curr_token) == 1:
                                curr_token = ''.join([curr_token, c])
                            else:
                                if curr_token[-1].isupper():
                                    token_list.append(curr_token[:-1])
                                    curr_type = set_type(c)
                                    curr_token = ''.join([curr_token[-1], c])
                                else:
                                    curr_token = ''.join([curr_token, c])
                    else:
                        if curr_type != "symbol" or (curr_type == "symbol" and len(curr_token) > 1):
                            if curr_type == "number":
                                token_list.append(str(int(curr_token)))
                            else:
                                token_list.append(curr_token)

                        curr_type = set_type(c)
                        curr_token = c
                elif c.isdigit():
                    if curr_type == "number":
                        curr_token = ''.join([curr_token, c])
                    else:
                        if curr_type != "symbol" or (curr_type == "symbol" and len(curr_token) > 1):
                            token_list.append(curr_token)

                        curr_type = set_type(c)
                        curr_token = c
                else:
                    if curr_type == "symbol":
                        if curr_token[0] == c:
                            curr_token = ''.join([curr_token, c])
                        else:
                            if len(curr_token) > 1:
                                token_list.append(curr_token)

                            curr_type = set_type(c)
                            curr_token = c
                    else:
                        if curr_type == "number":
                            token_list.append(str(int(curr_token)))
                        else:
                            token_list.append(curr_token)
                        curr_type = set_type(c)
                        curr_token = c

    if len(curr_token) != 0:
        if curr_type == "number":
            token_list.append(str(int(curr_token)))
        else:
            token_list.append(curr_token)

    row["Tokens"] = token_list
    return row


def check_english_tokens(row, manual_dict):
    token_list = row['Tokens']
    english_list = list()
    non_english_list = list()

    for token in token_list:
        lc_token = token.lower()

        if lc_token in nltk_dict \
                or lem.lemmatize(lc_token) in nltk_dict \
                or lem.lemmatize(lc_token, pos="v") in nltk_dict \
                or lem.lemmatize(lc_token, pos="a") in nltk_dict \
                or lem.lemmatize(lc_token, pos="s") in nltk_dict \
                or lem.lemmatize(lc_token, pos="r") in nltk_dict:
            if len(lc_token) > 3:
                english_list.append(lc_token)
            else:
                if lc_token not in manual_dict and lc_token not in stop_words:
                    non_english_list.append(lc_token)
                    english_list.append("_")
                else:
                    english_list.append(lc_token)
        else:
            non_english_list.append(lc_token)
            english_list.append("_")

    row["English_Tokens"] = english_list
    row["Not_English_Tokens"] = non_english_list
    all_non_english_tokens.update(non_english_list)

    return row


# Taken from https://stackoverflow.com/questions/24017363/how-to-test-if-one-string-is-a-subsequence-of-another
# check if x is a sub-sequence of y
def is_subseq(x, y):
    it = iter(y)
    return all(c in it for c in x)


def is_restricted_subseq(x, y):
    i = 0
    j = 0
    vowels = "aeiou"

    while i < len(x) and j < len(y):
        if x[i] == y[j]:
            if x[i] in vowels and i != 0 and j != 0 and x[i-1] != y[j-1]:
                pass
            else:
                i = i + 1
        j = j + 1

    if i == len(x):
        return True
    return False


def create_word_vector_from_context(context, model):
    # return the average word vector for each token in the context
    accumulated_vector = None
    context_length = 0

    for token in context:
        vector = model.wv[token] if token in model.wv and len(token) > 1 else None
        if vector is None:
            continue

        context_length = context_length + 1

        if accumulated_vector is None:
            accumulated_vector = vector
        else:
            accumulated_vector = accumulated_vector + vector

    if context_length == 0:
        print("Context found to be of zero length.")
        return None

    return accumulated_vector / context_length


def create_weighted_word_vector(context, model, weights, len_restriction):
    accumulated_vector = None

    for token in context:
        vector = model.wv[token] if token in model.wv and len(token) > len_restriction else None
        if vector is None:
            continue

        if accumulated_vector is None:
            accumulated_vector = vector * weights[token]
        else:
            accumulated_vector = accumulated_vector + vector * weights[token]

    if accumulated_vector is None:
        print("Context found to be of zero length. --candidate")
        return None

    return accumulated_vector


# From: https://www.geeksforgeeks.org/python-check-if-a-list-is-contained-in-another-list/
def find_matching_seq_index(A, B):
    if len(A) > len(B):
        return -1

    n = len(A)
    for i in range(len(B) - n + 1):
        if A == B[i:i + n]:
            return i

    return -1
    # return any(A == B[i:i + n] for i in range(len(B) - n + 1))


def find_word_context(tab, cols, word):
    word_list = []
    for col in cols:
        curr_col = tab[col]
        for sent in curr_col:
            if isinstance(sent, str):
                if word not in sent.lower():
                    continue

                curr_sent = [w for w in re.split("[;:'/. ,&()_-]", sent.lower()) if
                             len(w) > 1 and w not in stop_words]

                if " " in word:
                    acronym_words = [w for w in re.split("[ -]", word) if
                                     len(w) > 1 and w not in stop_words]  # Note: currently not considering stop words
                    seq_length = len(acronym_words)
                    seq_index = find_matching_seq_index(acronym_words, curr_sent)
                    while seq_index >= 0:
                        left_index = seq_index - 3 if seq_index - 3 >= 0 else 0
                        right_index = seq_index + seq_length + 3 if seq_index + seq_length + 3 <= len(curr_sent) \
                            else len(curr_sent)
                        word_list.extend(curr_sent[left_index:right_index])
                        curr_sent = curr_sent[right_index:]
                        seq_index = find_matching_seq_index(acronym_words, curr_sent)
                else:
                    while word in curr_sent:
                        word_index = curr_sent.index(word)
                        left_index = word_index - 3 if word_index - 3 >= 0 else 0
                        right_index = word_index + 3 if word_index + 3 <= len(curr_sent) else len(curr_sent)
                        word_list.extend(curr_sent[left_index:right_index])
                        curr_sent = curr_sent[right_index:]
    return word_list


def get_candidate_word_vector(cand, model, catalog, cols):
    if cand in cand_vectors:
        return cand_vectors[cand]

    desc_context = find_word_context(catalog, cols, cand)
    count = dict(Counter(desc_context))
    count = {k: v * (1 / len(desc_context)) for k, v in count.items()}
    vec = create_weighted_word_vector(list(set(desc_context)), model, count, 1)
    cand_vectors[cand] = vec

    return vec


def get_word_candidates(token, letter_dict):
    lc_letter = token[0].lower()
    if lc_letter not in letter_dict:
        return []

    all_words = letter_dict[lc_letter]
    return [w for w in all_words if len(token) < len(w) and is_subseq(token, w)]


def get_nearest_word(context_vector, word_candidates, model, catalog, cols):
    nearest_word = None
    best_score = -1
    word_list = []

    for word in word_candidates:
        if " " in word:
            acronym_words = re.split("[ ]", word)
            is_present = True
            for w in acronym_words:
                if w not in model.wv:
                    is_present = False
                    break
            word_vector = get_candidate_word_vector(word, model, catalog, cols) if is_present else None
        else:
            word_vector = get_candidate_word_vector(word, model, catalog, cols) if word in model.wv else None
        if word_vector is None:
            continue

        if context_vector is not None:
            curr_score = cosine_similarity(context_vector.reshape(1, -1), word_vector.reshape(1, -1))[0][0]
            if curr_score > best_score:
                nearest_word = word
                best_score = curr_score

            word_list.append((word, curr_score))
        else:
            word_list.append((word, 0.0))

    sorted_list = sorted(word_list, key=lambda x: x[1], reverse=True)
    return nearest_word, sorted_list


def normalize(token, context, model, cand_rules, letter_dict, catalog, cols):
    # create the context embedding
    context_vector = create_word_vector_from_context(context, model)

    # multiple-word abbrev/acronyms
    candidates = get_word_rules(token, cand_rules)

    # single-token abbrev
    word_candidates = get_word_candidates(token, letter_dict)
    word_candidates.extend(candidates)

    nearest_word, word_list = get_nearest_word(context_vector, word_candidates, model, catalog, cols)

    return context, nearest_word, word_list


def normalize_tokens(row, n, model, cand_rules, letter_dict, catalog, cols):
    nonenglish_list = row["Not_English_Tokens"]

    context_list = [token.lower() for token in row["Tokens"] if token[0].isalpha()]
    normalized_list = []
    all_scored_candidate = []
    top_n_candidates = []

    for token in nonenglish_list:
        if not token[0].isalpha():
            normalized_list.append(token)
            all_scored_candidate.append([(token, 0.0)])
            top_n_candidates.append([(token, 0.0)])
            continue

        context, normalized_word, word_list = normalize(token, context_list, model, cand_rules, letter_dict, catalog, cols)

        normalized_list.append(normalized_word)
        all_scored_candidate.append(word_list)
        top_n_candidates.append(word_list[:n])

    row["Token_Context"] = context_list
    row["Normalized_English_Tokens"] = normalized_list
    row["Candidate_Lists"] = all_scored_candidate
    row["Top_N_Candidates"] = top_n_candidates

    return row


def create_manual_dict(tab, cols):
    word_list = set()
    for col in cols:
        curr_col = tab[col]
        count = 0
        for word in curr_col:
            count = count + 1
            if isinstance(word, str):
                word_list.update([w.lower() for w in re.split("[/. ,&()_-]", word) if len(w) != 0])

    return word_list


def find_all_token_seq(tab, cols, all_letters, max_length):
    all_seq = []
    for col in cols:
        curr_col = tab[col]
        for text in curr_col:
            if isinstance(text, str):
                all_seq.extend([w.lower() for w in re.split("[.,;:()=*&]", text) if len(w) != 0])  # split into sentences or phrases

    all_token_seq = []
    for seq in all_seq:
        tokens = [w for w in re.split("[ ]", seq) if len(w) != 0] # split sent/phrase into words
        for i in range(len(tokens)):
            if tokens[i][0] not in all_letters or tokens[i] in stop_words:
                continue

            window_range = min(max_length, len(tokens) - i)
            all_token_seq.extend([" ".join(tokens[i:i + j + 1]) for j in range(window_range) if
                                  tokens[i:i + j + 1][-1] not in stop_words])  # does not end with a stopord

    return list(set(all_token_seq))


def find_candidate_rules(trie, token_seqs):
    all_rules = dict()

    for ts in token_seqs:
        prefixes = [{trie.root}] #prefixes[i] = F[i] in <LCS-based Candidate Rule Generation>
        for i in range(len(ts)):
            new_prefixes = set()
            letter = ts[i]
            if letter.isalpha():
                index = char_to_index(letter) # convert English letters to index: a = 0, b = 1, ..., z = 25
                for node in prefixes[i]:
                    if node.children[index]:
                        new_prefixes.add(node.children[index])

            new_prefixes = prefixes[i] | new_prefixes # "|": symmetric difference in set
            prefixes.append(new_prefixes)

        for node in prefixes[len(ts)]:
            if node.is_token:
                curr_rules = all_rules[node.token] if node.token in all_rules else []
                curr_rules.append(ts)
                all_rules[node.token] = curr_rules

    return all_rules


def check_rule_validity(token, rule):
    split_rule = re.split("[ ]", rule)
    i = 0
    for t in split_rule:
        if i >= len(token):
            return False

        if token[i] == t[0]:
            i = i + 1
        elif t in stop_words:
            continue
        else:
            return False

    if i != len(token):
        return False
    return True


def get_word_rules(token, cand_rules):
    if token not in cand_rules:
        return []

    rules = cand_rules[token]

    filtered_rules = []
    for r in rules:
        if len(token) < len(r) and is_subseq(token, r) and token[0] == r[0]:
            if check_rule_validity(token, r):
                filtered_rules.append(r)

    return filtered_rules

def create_inverted_index(tab, bus_term):
    index = {}
    for _, row in tab.iterrows():
        tokens = row['Tokens']
        bt = row[bus_term]
        for token in tokens:
            lc_token = token.lower()
            if lc_token not in stop_words and lc_token[0].isalpha():
                if lc_token not in index:
                    index[lc_token] = {bt}
                else:
                    index[lc_token].add(bt)
    return index


def combine_candidate_lists(row):
    english_list = row["English_Tokens"]
    non_english_list = row["Not_English_Tokens"]
    top_n_list = row["Top_N_Candidates"]
    combined_candidate_list = []

    i = 0
    for t in english_list:
        if t == "_":
            if len(top_n_list[i]) == 0:
                combined_candidate_list.append([(non_english_list[i], -1.0)])
            else:
                combined_candidate_list.append(top_n_list[i])
            i = i + 1
        else:
            combined_candidate_list.append([(t, 0.99)])

    row["Combined_Candidate_List"] = combined_candidate_list
    return row


def generate_derived_strings(list_of_lists):
    derived_tuples = list(itertools.product(*list_of_lists))
    derived_strings = []
    for tuples in derived_tuples:
        curr_str = []
        curr_score = 1.0
        has_score = False
        for t in tuples:
            curr_str.append(t[0])
            if t[1] != -1.0 and t[1] != 0.0:
                has_score = True
                curr_score = curr_score*t[1]

        if not has_score:
            curr_score = 0.0

        derived_strings.append((" ".join(curr_str), curr_score))

    sorted_derived_strings = sorted(derived_strings, key=lambda x: x[1], reverse=True)
    return [s[0] for s in sorted_derived_strings]
    # return sorted(derived_strings, key=lambda x: x[1], reverse=True)


def convert_to_list(tok_list):
    return [[t.lower() for t in tok_list]]


def lowercase_list(str_list):
    return [w.lower() for w in str_list if w != " "]


def update_candidate_lists(row, col_name, n, local_cnn_dict, global_cnn_dict):
    curr_col = row[col_name]
    english_list = row["English_Tokens"]
    non_english_list = row["Not_English_Tokens"]
    candidate_lists = row["Candidate_Lists"]

    expansion_dict = local_cnn_dict[curr_col] if curr_col in local_cnn_dict else None
    local = True
    if expansion_dict is None:
        expansion_dict = global_cnn_dict
        local = False

    i = 0
    top_n_candidates = []
    for token in english_list:
        if token == "_":
            non_eng_tok = non_english_list[i]
            tok_expansion_set = expansion_dict[non_eng_tok] if non_eng_tok in expansion_dict else set()
            used_expansion_set = set()
            candidates = candidate_lists[i]
            updated_candidates = []
            for cand in candidates:
                if cand[0] in tok_expansion_set:
                    updated_candidates.append((cand[0], cand[1] + 0.4) if local else (cand[0], cand[1] + 0.3))
                    used_expansion_set.add(cand[0])
                else:
                    updated_candidates.append(cand)

            unused_expansion_set = tok_expansion_set - used_expansion_set
            for exp in unused_expansion_set:
                updated_candidates.append((exp, 0.4) if local else (exp, 0.3))

            # add later
            # if len(updated_candidates) == 0:

            updated_candidates = sorted(updated_candidates, key=lambda x: x[1], reverse=True)
            candidate_lists[i] = updated_candidates
            top_n_candidates.append([t[0] for t in updated_candidates[:n]])

            i = i + 1

    row["Candidate_Lists"] = candidate_lists
    row["Top_N_Candidates"] = top_n_candidates
    return row

# taken from https://stackoverflow.com/questions/35596128/how-to-generate-a-word-frequency-histogram-where-bars-are-ordered-according-to
def plot_cryptic_token_hist(word_list):
    from collections import Counter
    import numpy as np
    import matplotlib.pyplot as plt

    # counts = Counter(word_list)
    n = 100
    counts = dict(Counter(word_list).most_common(n))

    labels, values = zip(*counts.items())

    # sort your values in descending order
    indSort = np.argsort(values)[::-1]

    # rearrange your data
    labels = np.array(labels)[indSort]
    values = np.array(values)[indSort]

    indexes = np.arange(len(labels))

    bar_width = 0.35

    plt.bar(indexes, values)

    # add labels
    plt.xticks(indexes + bar_width, labels, rotation=270)
    plt.show()


def run_cnn_initializer(catalog, tab_name, col_name, bus_term, descr):
    n = 3 # top n rewrite rules for each token 

    # a new csv file containing only the table or column names
    col_table_path = 'experiment_data/tc_names.csv'
    col_table = pd.read_csv(col_table_path).drop_duplicates()
    col_table = col_table.apply(lambda row: extract_tokens(row, col_name), axis=1).reset_index() #tokenize col_name
    col_table["index"] = col_table.index
    col_table = col_table.rename({"index": "col_id"}, axis=1)
    
    # a new csv file containing only the business terms and/or descriptions of the business terms
    bt_table_path = "experiment_data/bus_terms.csv"
    bt_table = pd.read_csv(bt_table_path)
    bt_table = bt_table.apply(lambda row: extract_tokens(row, bus_term), axis=1).reset_index()
    bt_table["index"] = bt_table.index
    bt_table = bt_table.rename({"index": "bt_id"}, axis=1)

    gold = catalog[[col_name, bus_term]]
    gold = gold[~gold[bus_term].isna()].drop_duplicates()
    cols = [bus_term, descr] if bus_term != descr else [descr]
    catalog = catalog[cols]

    inverted_index = create_inverted_index(bt_table, bus_term)

    # domain corpus: H_D
    manual_dict = create_manual_dict(catalog, cols)

    letter_dict = dict() # key: letter, value: list of words starting with the key letter in manual_dict
    for wrd in manual_dict:
        letter_list = letter_dict.get(wrd[0].lower(), [])
        letter_list.append(wrd)
        letter_dict[wrd[0].lower()] = letter_list

    all_token_letters = set()

    # detecting noisy tokens
    col_table = col_table.apply(lambda row: check_english_tokens(row, manual_dict), axis=1)

    trie = Trie()
    # insert all tokens t in U into Tri and calculate the global maximum length
    max_len = 0
    for t in all_non_english_tokens:
        if not t[0].isalpha():
            continue

        trie.insert(t)
        all_token_letters.add(t[0]) # stores the set of the first letters of all tokens
        max_len = max(max_len, min(len(t) + 5, len(t) * 2))
    # find all possible sequences that satisfies the pruning rules
    all_token_seq = find_all_token_seq(catalog, cols, all_token_letters, max_len)

    cand_rules = find_candidate_rules(trie, all_token_seq)

    bin_model_name = "gensim_fasttext_pretrained_bin_model.pickle"
    with open(bin_model_name, "rb") as f:
        model = pickle.load(f)
        print("success")

    col_table = col_table.apply(lambda row: normalize_tokens(row, n, model, cand_rules,
                                                             letter_dict, catalog, cols), axis=1)

    col_table = col_table.apply(lambda row: combine_candidate_lists(row), axis=1)

    col_table["Derived_Strings"] = col_table["Combined_Candidate_List"].map(generate_derived_strings)
    col_table["Updated"] = True

    print("done")

    return col_table[["col_id", col_name, "Tokens", "Combined_Candidate_List", "Derived_Strings", "Updated"]], \
           bt_table, gold, inverted_index


nltk_dict = set(words.words())
stop_words = set(stopwords.words('english'))
lem = WordNetLemmatizer()

all_non_english_tokens = set()
cand_vectors = dict()
