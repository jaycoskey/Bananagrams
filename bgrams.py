#!/usr/bin/python3

import argparse
import collections
from functools import reduce
from itertools import groupby
import math
import multiprocessing  # cpu_count, Pool
from operator import mul
import random
import re
import string
import sys
import time

# Added a limit of 250K pairs after hitting a MemoryError with ~290K  pairs (and ~1150 singletons) on 8GB of RAM.
# (The highest load is in the computation of triples, and pair count is a better predictor of that than singleton count.)
MAX_PAIR_COUNT = 250000

PATH_LINUX_WORDS = '/usr/share/dict/linux.words'
PATH_LOG_WORDLISTS = './bgrams.log'
PATH_WORDS9 = './bgrams9.txt'
POOL_SIZE = 3
TARGET_HISTO = collections.Counter({
    'a': 13, 'b': 3, 'c': 3, 'd': 6, 'e': 18,
    'f': 3, 'g': 4, 'h': 3, 'i': 12, 'j': 2,
    'k': 2, 'l': 5, 'm': 3, 'n': 8, 'o': 11,
    'p': 3, 'q': 2, 'r': 9, 's': 6, 't': 9,
    'u': 6, 'v': 3, 'w': 3, 'x': 2, 'y': 3,
    'z': 2
})
TARGET_WORDCOUNT = 16
TARGET_WORDLENGTH = 9
TUPLE_CHUNK_SIZE = 10000
TUPLE_PROGRESS_CHARS = ['a', 'b', 'c']

# Given equal loads, the first pool starts and ends earlier (due to IO?), so give it more tuples to handle.
# Would need to experiment to determine the optimal weighting.
TUPLE_WEIGHTS = [50, 35, 20]
TUPLE_WORDCOUNT = 5

PATH_LOG_WORDLISTS_TEST = './bgrams_test.log'
PATH_WORDS9_TEST = './bgrams9_test.txt'
TARGET_HISTO_TEST = collections.Counter({
    'a': 12, 'b': 2, 'c': 4, 'd': 5, 'e': 5,
    'f': 0, 'g': 0, 'h': 0, 'i': 5, 'j': 1,
    'k': 1, 'l': 1, 'm': 0, 'n': 1, 'o': 1,
    'p': 0, 'q': 1, 'r': 2, 's': 2, 't': 4,
    'u': 3, 'v': 2, 'w': 0, 'x': 1, 'y': 0,
    'z': 1
})
TARGET_WORDCOUNT_TEST = 6


class WordCache():
    def __init__(self, words):
        self._primes = self._primes_less_than(102)
        self._alpha2prime = self._get_alpha2prime(0, 1)
        words_sorted = sorted(words, key=lambda w: self._word2hash(w))
        self._hash2anagrams = {}
        for hash, anagroup in groupby(words_sorted, self._word2hash):
            anagrams = list(anagroup)
            self._hash2anagrams[hash] = anagrams

    # From O'Reilly's Python Cookbook
    def _primes_less_than(self, n):
        aux = { }
        return [aux.setdefault(p, p) for p in range(2, n)
                if 0 not in [p % d for d in aux if p >= d + d]]

    def _get_alpha2prime(self, p_init, p_step):
        alphabet = string.ascii_lowercase[:26]
        alpha2prime = {}
        for i in range(0, 26):
            alpha2prime[alphabet[i]] = self._primes[p_init + p_step * i]
        return alpha2prime

    def _counter2hash(self, counter: collections.Counter):
        return reduce(mul, [self._alpha2prime[c] ** v for c, v in counter.items()]) % 1000000000

    def _word2hash(self, word: str):
        return reduce(mul, [self._alpha2prime[c] for c in word]) % 1000000000

    def get_anagrams_by_hash(self, hash):
        return self._hash2anagrams.get(hash, [])

    def get_anagrams_by_counter(self, counter: collections.Counter):
        assert(isinstance(counter, collections.Counter))
        return self._hash2anagrams.get(self._counter2hash(counter), [])


def add_missing_letters(w_j, w_k, w_q, w_x, w_z, letters, words):
    if 'j' in letters:
        words.append(get_rand_item(w_j))
    if 'k' in letters:
        words.append(get_rand_item(w_k))
    if 'q' in letters:
        words.append(get_rand_item(w_q))
    if 'x' in letters:
        words.append(get_rand_item(w_x))
    if 'z' in letters:
        words.append(get_rand_item(w_z))


def adjust_word_count(all_words, target_count, words):
    shortfall = target_count - len(words)
    if shortfall > 0:  # Add words
        for i in range(0, shortfall):
            words.append(get_rand_item(all_words))
    elif shortfall < 0:  # Remove words
        for i in range(0, -shortfall):
            words.pop()


def chunk_by_weight(items, weights):
    item_count = len(items)
    result = []
    begin_index = 0
    for i in range(0, len(weights)):
        if i == len(weights) - 1:
            step_size = len(items) - begin_index
        else:
            step_size = math.floor(weights[i] * item_count)
        result.append(items[begin_index: begin_index + step_size])
        begin_index += step_size
    return result


def chunkify(items, number_of_bins):
    chunk_size = math.ceil(len(items) / number_of_bins)
    return chunks_of(items, chunk_size)


def chunks_of(items, chunk_size):
    return [items[i: i + chunk_size] for i in range(0, len(items), chunk_size)]


def constraint_satisfaction_gap(constraint, counter):
    gap_counter = constraint.copy()
    gap_counter.subtract(counter)
    if any(gap < 0 for gap in gap_counter.values()):
        return None
    else:
        return gap_counter


def constraint_satisfaction_gap_from_wordlist(constraint, words):
    counter = collections.Counter()
    for w in words:
        counter += collections.Counter(w)
    return constraint_satisfaction_gap(constraint, counter)


def flatten(lsts):
    # [item for lst in lsts for item in lst]
    result = []
    for lst in lsts:  # Or reversed(lsts)
        result[0:0] = lst
    return result


def get_gap_histo(words):
    histograms = list(map(lambda w: collections.Counter(w), words))
    histo = collections.Counter()
    for h in histograms:
        histo += h
    result = TARGET_HISTO.copy()
    result.subtract(histo)
    return result


def get_histo_map(words):
    histo_map = {}
    for w in words:
        histo_map[collections.Counter(w).items()] = w
    return histo_map


def get_initial_list(all_words, w_j, w_k, w_q, w_z, w_x, wordcount):
    rand_j0 = get_rand_item(w_j)
    rand_k0 = get_rand_item(w_k)
    rand_q0 = get_rand_item(w_q)
    rand_x0 = get_rand_item(w_x)
    rand_z0 = get_rand_item(w_z)

    rand_j1 = get_rand_item(w_j)
    rand_k1 = get_rand_item(w_k)
    rand_q1 = get_rand_item(w_q)
    rand_x1 = get_rand_item(w_x)
    rand_z1 = get_rand_item(w_z)

    rand_0 = get_rand_item(all_words)
    rand_1 = get_rand_item(all_words)
    rand_2 = get_rand_item(all_words)
    rand_3 = get_rand_item(all_words)
    rand_4 = get_rand_item(all_words)
    rand_5 = get_rand_item(all_words)
    words = [rand_j0, rand_k0, rand_q0, rand_x0, rand_z0,
             rand_j1, rand_k1, rand_q1, rand_x1, rand_z1,
             rand_0, rand_1, rand_2, rand_3, rand_4, rand_5]
    return words[0:wordcount]


def get_rand_item(words) -> str:
    return random.choice(words)


def get_score(words) -> int:
    word_count = len(words)
    gap_histo = get_gap_histo(words)
    if any(gap < 0 for gap in gap_histo.values()):
        result = 0
    else:
        result = word_count
    return result


def get_singletons(constraint, words):
    singletons = [(w, collections.Counter(w)) for w in words
                  if is_constraint_satisfied(constraint, collections.Counter(w))]
    return singletons


def get_datestamp():
    return time.strftime('[%Y-%m-%d @ %H:%M:%S]')


def get_tuples2_from_singletons(constraint, singletons1, singletons2):
    pairs = [(w1, w2, c1 + c2)
             for (w1, c1) in singletons1
             for (w2, c2) in singletons2
             if w2 > w1 and is_constraint_satisfied(constraint, c1 + c2)]
    return pairs


def get_tuples3_from_tuples2(constraint, singletons, tuples2, progress_char):
    tuple2_chunks = chunks_of(tuples2, TUPLE_CHUNK_SIZE)
    triple_chunks = []
    for tuple2_chunk in tuple2_chunks:
        triple_chunks.append(
            [(w1, w2, w3, c12 + c3)
             for (w1, w2, c12) in tuple2_chunk
             for (w3, c3) in singletons
             if w3 > w2 and is_constraint_satisfied(constraint, c12 + c3)]
        )
        if len(tuple2_chunk) == TUPLE_CHUNK_SIZE:
            show_progress(progress_char.upper())
        elif len(tuple2_chunk) < TUPLE_CHUNK_SIZE:
            show_progress(progress_char.lower())
        else:
            show_progress('?')
    triples = flatten(triple_chunks)
    return triples


def get_tuples4_from_tuples3(constraint, singletons, tuples3, progress_char):
    tuple3_chunks = chunks_of(tuples3, TUPLE_CHUNK_SIZE)
    quadruple_chunks = []
    for tuple3_chunk in tuple3_chunks:
        quadruple_chunks.append(
            [(w1, w2, w3, w4, c123 + c4)
             for (w1, w2, w3, c123) in tuple3_chunk
             for (w4, c4) in singletons
             if w4 > w3 and is_constraint_satisfied(constraint, c123 + c4)]
        )
        if len(tuple3_chunk) == TUPLE_CHUNK_SIZE:
            show_progress(progress_char.upper())
        elif len(tuple3_chunk) < TUPLE_CHUNK_SIZE:
            show_progress(progress_char.lower())
        else:
            show_progress('?')
    quadruples = flatten(quadruple_chunks)
    return quadruples


def get_tuples5_from_tuples4(constraint, singletons, tuples4, word_cache):
    # Old version:
    # quadruples = tuples4
    # quintuples = [(w1, w2, w3, w4, w5)  # Do not return counters; they're no longer needed.
    #               for (w1, w2, w3, w4, c1234) in quadruples
    #               for (w5, c5) in singletons
    #               if w5 > w4 and is_constraint_satisfied(constraint, c1234 + c5)]
    # last_items = [w1, w2, w3, w4, c123 + c4)
    # return quintuples
    #
    # More efficient, cache-enabled version:
    quintuples = []
    for (w1, w2, w3, w4, c1234) in tuples4:
        # Direct subtraction doesn't track negative counts, but that's OK.
        for w5 in word_cache.get_anagrams_by_counter(constraint - c1234):
            if w5 in singletons:
                quintuples.append((w1, w2, w3, w4, w5))
    return quintuples


def get_quintuples_from_wordlist(words9, word_cache, constraint):
    show_progress('\n' + get_datestamp() + ' ')

    singletons = get_singletons(constraint, words9)
    show_progress('{0}'.format(len(singletons)))

    # singleton_chunks = chunkify(singletons, multiprocessing.cpu_count() - 1)
    singleton_chunks = chunk_by_weight(singletons, normalize(TUPLE_WEIGHTS))
    singleton_chunks_with_progress_chars = list(zip(singleton_chunks, TUPLE_PROGRESS_CHARS))

    # show_progress('(')
    # for (chunk, prog_char) in singleton_chunks_with_progress_chars:
    #     print('{0:s}:{1:d} '.format(prog_char, len(chunk)), end='')
    # show_progress(')')

    # Below is the async version of: pairs = get_tuples2_from_singletons(constraint, singletons)
    pair_chunk_asyncs = [pool.apply_async(get_tuples2_from_singletons, (constraint, singleton_chunk, singletons))
                         for singleton_chunk in singleton_chunks]
    pair_chunks = [pair_chunk.get(timeout=None) for pair_chunk in pair_chunk_asyncs]
    pairs = flatten(pair_chunks)
    if len(pairs) > MAX_PAIR_COUNT:
        print('\tAborting search for this wordlist.  Candidate pair count ({0}) exceeds max setting ({1}).'
              .format(len(pairs), MAX_PAIR_COUNT)
              )
        pairs = []
    show_progress(', {0}:'.format(len(pairs)))

    # Below is the async version of: triples = get_tuples3_from_tuples2(constraint, singletons, pairs)
    triple_chunk_asyncs = [pool.apply_async(get_tuples3_from_tuples2,
                                            (constraint, singleton_chunk, pairs, prog_char))
                           for (singleton_chunk, prog_char) in singleton_chunks_with_progress_chars]
    triple_chunks = [triple_chunk.get(timeout=None) for triple_chunk in triple_chunk_asyncs]
    triples = flatten(triple_chunks)
    show_progress(', {0}:'.format(len(triples)))

    # Below is the async version of: quadruples = get_tuples4_from_tuples3(constraint, singletons, triples)
    quadruple_chunk_asyncs = [pool.apply_async(get_tuples4_from_tuples3,
                                               (constraint, singleton_chunk, triples, prog_char))
                              for (singleton_chunk, prog_char) in singleton_chunks_with_progress_chars]
    quadruples = flatten([quadruple_chunk.get(timeout=None) for quadruple_chunk in quadruple_chunk_asyncs])
    show_progress(', {0}'.format(len(quadruples)))

    # Caching saves us some time searching for the last item in e list.
    quintuples = get_tuples5_from_tuples4(constraint, singletons, quadruples, word_cache)
    show_progress(', {0}'.format(len(quintuples)))
    return quintuples

def get_words_with_char(words, char):
    return [w for w in words if re.search(char, w)]


def is_constraint_satisfied(constraint, counter):
    gap_counter = constraint.copy()
    gap_counter.subtract(counter)
    if any(gap < 0 for gap in gap_counter.values()):
        return False
    else:
        return True


def log_wordlist(path_log_wordlists, words):
    msg_log = '{0:d}: {1:s}'.format(len(words), words)
    with open(path_log_wordlists, 'a') as f:
        f.write(get_datestamp() + ' ' + msg_log + '\n')


def normalize(weights):
    total = sum(weights)
    normalized = [w/(1.0*total) for w in weights]
    return normalized


def print_wordlist(header, words):
    print(header)
    for w in words:
        print('\t{0:s}'.format(w))


def read_words9(path_words9):
    with open(path_words9, 'r') as f:
        lines = [line.rstrip('\n') for line in f]
        w9 = [line for line in lines if line.islower() and len(line) == TARGET_WORDLENGTH]
    return w9


def remove_excess_letters(letters, words):
    for c in letters:
        is_found = False
        for w in words:
            if c in w:
                words.remove(w)
                is_found = True
                break
            if is_found:
                break


def show_progress(txt):
    print(txt, end='', flush=True)


def write_words9(do_test) -> None:
    if do_test:
        w9 = ['aardvarks', 'abdicated', 'acquaints', 'actualize', 'aboideaux', 'adjective']
    else:
        with open(PATH_LINUX_WORDS) as f:
            lines = [line.rstrip('\n') for line in f]
            w9 = [line for line in lines
                  if line.islower() and len(line) == TARGET_WORDLENGTH]
    with open(PATH_WORDS9, 'w') as w9f:
        for word in w9:
            w9f.write(word + '\n')


def main(pool, parser_args):
    if parser_args.test:
        path_log_wordlists = PATH_LOG_WORDLISTS_TEST
        path_words9 = PATH_WORDS9_TEST
        target_histo = TARGET_HISTO_TEST
        target_wordcount = TARGET_WORDCOUNT_TEST
    else:
        path_log_wordlists = args.log_file.name
        path_words9 = args.input_file.name
        target_histo = TARGET_HISTO
        target_wordcount = TARGET_WORDCOUNT

    # If there is not a wordlist available with only 9-letter words,
    # then call write_words9() to create one.
    words9 = read_words9(path_words9)
    word_cache = WordCache(words9)
    w_j = get_words_with_char(words9, 'j')
    w_k = get_words_with_char(words9, 'k')
    w_q = get_words_with_char(words9, 'q')
    w_x = get_words_with_char(words9, 'x')
    w_z = get_words_with_char(words9, 'z')
    if args.verbose:
        print("words9 count = {0:d}".format(len(words9)))
        print("words_j count = {0:d}".format(len(w_j)))
        print("words_k count = {0:d}".format(len(w_k)))
        print("words_q count = {0:d}".format(len(w_q)))
        print("words_x count = {0:d}".format(len(w_x)))
        print("words_z count = {0:d}".format(len(w_z)))

    wordlist0 = get_initial_list(words9, w_j, w_k, w_q, w_x, w_z,
                                 wordcount=target_wordcount - TUPLE_WORDCOUNT)
    if args.verbose:
        print_wordlist('Initial candidate word list:', wordlist0)

    cur_wordlist = wordlist0
    iter_count = 0
    is_full_wordlist_found = False
    while not is_full_wordlist_found:
        iter_count += 1
        constraint = constraint_satisfaction_gap_from_wordlist(target_histo, cur_wordlist)
        is_cur_wordlist_valid = constraint is not None
        if is_cur_wordlist_valid:
            quintuples = get_quintuples_from_wordlist(words9, word_cache, constraint)
            if len(quintuples) > 0:
                for (w1, w2, w3, w4, w5) in quintuples:
                    full_list = cur_wordlist + [w1, w2, w3, w4, w5]
                    log_wordlist(path_log_wordlists, full_list.__str__())
                    print_wordlist('\n{0:s} Full {1:d}-word wordlist found!'.format(get_datestamp(), target_wordcount),
                                   full_list)
                is_full_wordlist_found = True

        if not is_full_wordlist_found:
            show_progress('.')
            if is_cur_wordlist_valid:
                # If we already searched for a complementary quintuple, then start over again.
                cur_wordlist = get_initial_list(words9, w_j, w_k, w_q, w_x, w_z,
                                                wordcount=target_wordcount - TUPLE_WORDCOUNT)
            else:
                # If it has too many of some letters, then try a "minor" adjustment and continue.
                excess_letters = [k for k, v in get_gap_histo(cur_wordlist).items() if v < 0]
                if len(excess_letters) > 0:
                    remove_excess_letters(excess_letters, cur_wordlist)

                missing_letters = [k for k, v in get_gap_histo(cur_wordlist).items() if v > 0]
                if len(missing_letters) > 0:
                    add_missing_letters(w_j, w_k, w_q, w_x, w_z, missing_letters, cur_wordlist)
                adjust_word_count(words9, target_wordcount - TUPLE_WORDCOUNT, cur_wordlist)


if __name__ == '__main__':
    # Early configuration sanity check
    assert(POOL_SIZE == len(TUPLE_PROGRESS_CHARS))
    assert(POOL_SIZE == len(TUPLE_WEIGHTS))

    parser = argparse.ArgumentParser(prog='bgrams')
    parser.add_argument('-i', '--input_file',
                        default=PATH_WORDS9,
                        type=argparse.FileType('r'),
                        help='specify name of input file containing nine-letter words')
    parser.add_argument('-l', '--log_file',
                        default=PATH_LOG_WORDLISTS,
                        type=argparse.FileType('a'),
                        help='specify name of logfile')
    parser.add_argument('-t', '--test',
                        action='store_true',
                        help='run test version of script')
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='run in verbose mode')
    args = parser.parse_args(sys.argv[1:])
    pool = multiprocessing.Pool(processes=POOL_SIZE)
    try:
        main(pool, args)
    finally:
        pool.close()
        pool.join()
