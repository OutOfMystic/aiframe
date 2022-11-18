import itertools
import os
import random
from itertools import product

import Levenshtein
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

ROW_ADD_PROP = 0.2
ROW_REVERSE = True
ROW_REPEAT = 1
ROW_REPLACE_PROP = 0.4
WORD_ADD_PROP = 0.05
WORD_DOUBLE_ADD_PROP = 0.02


def load_real_input_words():
    path = os.path.join('examples', 'rnn', 'input_data', 'initial_data.csv')
    with open(path, 'r') as f:
        raw_data = f.read()
    rows = [load_row(row) for row in raw_data.split('\n') if load_row(row)]
    return rows


def load_synthetic_input_words(count=10000, words=()):
    synthesizeds = []
    for _ in range(count):
        word_numbers = random.randint(2, 6)
        synthesized = []
        for _ in range(word_numbers):
            if random.random() < 0.08:
                randnum = str(random.randint(0, 1000))
                synthesized.append(randnum)
            else:
                randword = random.choice(words)
                synthesized.append(randword)
        synthesizeds.append(synthesized)
    return synthesizeds


def load_row(row):
    symbolless = ''
    for char in row:
        if char == '-':
            symbolless += ' '
        elif char.isalpha() or char.isdigit() or char == ' ':
            symbolless += char
    words = [elem for elem in symbolless.split(' ') if elem]
    return words


def word_add_symbols(word):
    new_word = ''
    for char in word:
        chance = random.random()
        if chance < WORD_ADD_PROP:
            new_word += random.choice(all_symbols)
        if chance < WORD_DOUBLE_ADD_PROP:
            new_word += random.choice(all_symbols)
        new_word += char
    return new_word


def row_add_words(row, words):
    new_row = []
    for word in row:
        while random.random() < ROW_ADD_PROP:
            new_word = random.choice(words)
            new_row.append(new_word)
        new_row.append(word)
    return new_row


def get_pairs(rows, words):
    pairs = list()
    for row in rows:
        new_pairs = process_row(row, words)
        pairs.extend(new_pairs)
    return list(set(pairs))


def process_row(row, words):
    pairs = []
    for _ in range(ROW_REPEAT):
        new_row = row_add_words(row, words)
        for word_num, word in enumerate(new_row):
            new_row[word_num] = word_add_symbols(word)
        while random.random() < ROW_REPLACE_PROP:
            elem1 = random.choice(new_row)
            index1 = new_row.index(elem1)
            elem2 = random.choice(new_row)
            index2 = new_row.index(elem2)
            new_row[index1] = elem2
            new_row[index2] = elem1
        pair = (' '.join(row), ' '.join(new_row),)
        pairs.append(pair)
        if ROW_REVERSE:
            pair = (' '.join(new_row), ' '.join(row),)
            pairs.append(pair)
    return pairs


def format_input_data(pairs, dif_func, translate_func):
    first_inputs = [pair[0] for pair in pairs]
    second_inputs = [pair[1] for pair in pairs]
    to_tokenize = first_inputs + second_inputs
    indexes = [index for index in range(len(pairs))]

    tokenizer = Tokenizer(num_words=100, char_level=True)
    tokenizer.fit_on_texts(to_tokenize)
    #print(list(tokenizer.word_index.items()))

    input_data = []
    results = []
    for first, second, ind in zip(first_inputs, second_inputs, indexes):
        input_row = dif_func(first, second, tokenizer, translate_func)
        input_data.append(input_row)
        results.append([1, 0])
        #print(first, '|||', second, [1, 0])

        inds_to_choose = indexes.copy()
        inds_to_choose.remove(ind)
        chosen_ind = random.choice(inds_to_choose)
        bad_second = second_inputs[chosen_ind]
        while abs(len(bad_second) - len(first)) > 10:
            chosen_ind = random.choice(inds_to_choose)
            bad_second = second_inputs[chosen_ind]
        input_row = dif_func(first, bad_second, tokenizer, translate_func)
        input_data.append(input_row)
        results.append([0, 1])
        #print(first, '|||', bad_second, [0, 1])
    return input_data, results, tokenizer


def human_to_bot(first, second, tokenizer, translate_func):
    first = translate_func(first, tokenizer)
    second = translate_func(second, tokenizer)
    concatted = np.concatenate([first, second])
    return concatted


def differences(string1, string2, tokenizer=None, *args):
    max_words = 20
    levenshtein_grid = np.zeros((max_words, max_words,), dtype=np.float32)

    # solving words
    words1 = string1.split(' ')[:20]
    words2 = string2.split(' ')[:20]
    iterator = product(words1, words2)
    for i, words in enumerate(iterator):
        word1, word2 = words
        row = i // len(words2)
        col = i % len(words2)
        distance = Levenshtein.distance(word1, word2)
        levenshtein_grid[row, col] = 1 / (distance + 1)

    levenshtein_grid = levenshtein_grid.reshape((max_words ** 2,))
    matches1 = str_to_wordstats(string1)
    matches2 = str_to_wordstats(string2)
    concatted = np.concatenate([levenshtein_grid, matches1, matches2])
    return concatted


def str_to_tokens(string, tokenizer):
    sequences = tokenizer.texts_to_sequences([string])
    padded_seq = pad_sequences(sequences, maxlen=200)[0]
    return padded_seq


def str_to_one_hot(string, tokenizer):
    input_size = 75
    output_size = 10

    divider = input_size // output_size
    matrix = np.zeros((200, output_size, ))

    sequences = tokenizer.texts_to_sequences([string])
    padded_seq = pad_sequences(sequences, maxlen=200)[0]

    for i, mark in enumerate(padded_seq):
        mark_index = mark // (divider + 1)
        mark_value = mark % (divider + 1) / divider
        matrix[i, mark_index] = mark_value
    return matrix


def str_to_wordstats(string, tokenizer=None):
    sequences = [ords.index(ord(char)) for char in string.lower()]
    matches = np.zeros((75,))
    for char in sequences:
        matches[char] += 1
    matches /= 20
    for i in range(matches.shape[0]):
        matches[i] = min(matches[i], 1)
    return matches


def tokens2word(tokens):
    words = []
    last_word = []
    for token in tokens:
        if token == 1:
            words.append(last_word)
            last_word = []
        else:
            last_word += token
    return words


ord_iters = [[46, 32, 1104, 1105], range(48, 58), range(97, 123), range(1072, 1104)]
ords = list(itertools.chain(*ord_iters))
all_symbols = [chr(code) for code in ords]
words_path = os.path.join('examples', 'rnn', 'input_data', 'russian.txt')
with open(words_path, 'r') as f:
    ru_words = [word for word in f.read().split('\n') if word.isalpha()]
words_path = os.path.join('examples', 'rnn', 'input_data', 'english.txt')
with open(words_path, 'r') as f:
    eng_words = [word for word in f.read().split('\n') if word.isalpha()]


if __name__ == '__main__':
    rows = load_synthetic_input_words(100, words=ru_words)
    pairs = get_pairs(rows, ru_words)
    rows = load_synthetic_input_words(100, words=eng_words)
    pairs += get_pairs(rows, eng_words)
    pairs, results, tokenizer = format_input_data(pairs, differences, str_to_one_hot)
    print(pairs[0], results[0])
    print(pairs[1], results[1])
