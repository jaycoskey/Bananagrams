#!/usr/bin/python3

import argparse
import collections
from functools import reduce
import math
import multiprocessing  # cpu_count, Pool
from operator import mul
from os import listdir
from os.path import isfile, join
import random
import string
import sys
import time

CACHE_DIR = 'cache'
CACHE_FILE_BASENAME = 'cache_'
PATH_DICT = '/usr/share/dict/linux.words'
PATH_LOG_PROGRESS = './bgrams_12x12_progress.log'
PATH_LOG_SOLUTIONS = './bgrams_12x12_solutions.log'
PATH_WORDS12 = './bgrams12.txt'
POOL_SIZE = 3

BANANAGRAM_TILES = collections.Counter({
    'a': 13, 'b': 3, 'c': 3, 'd': 6, 'e': 18,
    'f': 3, 'g': 4, 'h': 3, 'i': 12, 'j': 2,
    'k': 2, 'l': 5, 'm': 3, 'n': 8, 'o': 11,
    'p': 3, 'q': 2, 'r': 9, 's': 6, 't': 9,
    'u': 6, 'v': 3, 'w': 3, 'x': 2, 'y': 3,
    'z': 2
})
COMMON_TILES = collections.Counter({
    'a': 13, 'b': 3, 'c': 3, 'd': 6, 'e': 18,
    'f': 3,  'g': 4, 'h': 3, 'i': 12,
             'l': 5, 'm': 3, 'n': 8, 'o': 11,
    'p': 3,          'r': 9, 's': 6, 't': 9,
    'u': 6,  'v': 3,                 'y': 3,
})
RARE_TILES = collections.Counter({
    'j': 2, 'k': 2, 'q': 2, 'x': 2, 'z': 2,
    'w': 3
})

# TARGET_WORDCOUNT = 12
WORDLENGTH = 12
# TUPLE_CHUNK_SIZE = 10000
# TUPLE_PROGRESS_CHARS = ['a', 'b', 'c']

# Given equal loads, the first pool starts and ends earlier (due to IO?), so give it more tuples to handle.
# Would need to experiment to determine the optimal weighting.
# TUPLE_WEIGHTS = [50, 35, 20]
# TUPLE_WORDCOUNT = 5


class WordlistCache:
    def __init__(self):
        self._primes = self._primes_less_than(102)
        # This value for _mod supports non-colliding hash
        # for 9 & 12-letter words.
        self._mod = 1000000000000000  # 10^15 < Integer.MAX_VALUE (64 bits)
        self._alpha2prime = self._get_alpha2prime(0, 1)
        self._hashset = set()

    def add(self, wordlist, do_cache):
        hash = self._wordlist2hash(wordlist)
        self._hashset.add(hash)
        if do_cache:
            path = self._hash2path(hash)
            with open(path, 'a+') as cache_file:
                cache_file.write('{0:d} {1:s}\n'.format(hash, wordlist.__str__()))

    def counter2hash(self, counter: collections.Counter):
        return reduce(mul, [self._alpha2prime[c] ** v for c, v in counter.items()]) % self._mod

    def has_hash_of_counter(self, counter):
        return self.has_hash(self.counter2hash(counter))

    def has_hash(self, hash):
        return hash in self._hashset

    def key_count(self):
        return len(self._hashset)

    def load_keys(self):
        cache_paths = [join(CACHE_DIR, file) for file in listdir(CACHE_DIR)
                       if isfile(join(CACHE_DIR, file))]
        cache_files_read = 0
        for cache_path in cache_paths:
            with open(cache_path, 'r') as cache_file:
                lines = cache_file.readlines()
                for line in lines:
                    parts = line.split(' ', 1)
                    line_hash = parts[0]
                    self._hashset.add(int(line_hash))
            cache_files_read += 1
            if cache_files_read % 1000 == 0:
                show_progress('C')
            elif cache_files_read % 100 == 0:
                show_progress('c')
        print('')

    def lookup_hash(self, target_hash):
        result = []
        path = self._hash2path(target_hash)
        with open(path, 'r') as cache_file:
            lines = cache_file.readlines()
            for line in lines:
                parts = line.split(' ', 1)
                line_hash = parts[0]
                # wordlist = parts[1]
                if int(line_hash) == target_hash:
                    result.append(line)
        return result

    @staticmethod
    def _hash2path(hash):
        return CACHE_DIR + '/' + CACHE_FILE_BASENAME + str(hash % 10000).zfill(4)

    @staticmethod
    def _primes_less_than(n):
        aux = {}
        # From O'Reilly's Python Cookbook
        return [aux.setdefault(p, p) for p in range(2, n)
                if 0 not in [p % d for d in aux if p >= d + d]]

    def _get_alpha2prime(self, p_init, p_step):
        alphabet = string.ascii_lowercase[:26]
        alpha2prime = {}
        for i in range(0, 26):
            alpha2prime[alphabet[i]] = self._primes[p_init + p_step * i]
        return alpha2prime

    def _wordlist2hash(self, wordlist):
        chars = ''.join(wordlist)
        return reduce(mul, [self._alpha2prime[c] for c in chars]) % self._mod


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


def constraint_satisfaction_gap_from_counter(constraint, counter):
    gap_counter = constraint.copy()
    gap_counter.subtract(counter)
    return gap_counter


def constraint_satisfaction_gap_from_wordlist(constraint, words):
    counter = get_counter_from_wordlist(words)
    return constraint_satisfaction_gap_from_counter(constraint, counter)


def flatten(lsts):
    # [item for lst in lsts for item in lst]
    result = []
    for lst in lsts:  # Or reversed(lsts)
        result[0:0] = lst
    return result


def get_datestamp():
    return time.strftime('[%Y-%m-%d @ %H:%M:%S]')


def get_initial_wordlist8(w_j, w_k, w_q, w_z, w_x, w_w):
    rand_j0 = get_rand_item(w_j)
    rand_k0 = get_rand_item(w_k)
    rand_q0 = get_rand_item(w_q)
    rand_x0 = get_rand_item(w_x)
    rand_z0 = get_rand_item(w_z)
    rand_w0 = get_rand_item(w_w)

    rand_j1 = get_rand_item(w_j)
    rand_k1 = get_rand_item(w_k)
    rand_q1 = get_rand_item(w_q)
    rand_x1 = get_rand_item(w_x)
    rand_z1 = get_rand_item(w_z)
    rand_w1 = get_rand_item(w_w)

    wordlist = [rand_j0, rand_k0, rand_q0, rand_x0, rand_z0, rand_w0,
                rand_j1, rand_k1, rand_q1, rand_x1, rand_z1, rand_w1
                ]
    # Remove a random selection of 4
    i0 = random.randint(0, 11)
    i1 = random.randint(0, 10)
    i2 = random.randint(0, 9)
    i3 = random.randint(0, 8)
    del wordlist[i0]
    del wordlist[i1]
    del wordlist[i2]
    del wordlist[i3]
    return wordlist # [:wordcount]


def get_counter_from_wordlist(words):
    counter = collections.Counter()
    for w in words:
        counter += collections.Counter(w)
    return counter


def get_rand_item(words) -> str:
    return random.choice(words)


def get_valid_wordlist4s(w_others, num_wordlists):
    valid_wordlist4s = []
    while len(valid_wordlist4s) < num_wordlists:
        wordlist4 = [get_rand_item(w_others), get_rand_item(w_others),
                     get_rand_item(w_others), get_rand_item(w_others)]
        cntr4 = get_counter_from_wordlist(wordlist4)
        gap8 = constraint_satisfaction_gap_from_counter(COMMON_TILES, cntr4)
        if all(gap_values >= 0 for gap_values in gap8.values()):
            valid_wordlist4s.append(wordlist4)
    return valid_wordlist4s


# The code for this function isn't pretty, but it doesn't need to be.
# It returns an 8-word list that ideally contains the right number of
#     j's, k's, q's, x's, z's (2 each), and w's (3)
# The caller checks to see if the wordlist return is actually usable.
def get_wordlist8(w_j, w_k, w_q, w_x, w_z, w_w,
                  w_k1, w_z1, w_w1,
                  w_k2, w_z2, w_w2,
                  w_jz, w_kq, w_kw, w_qz, w_wz, w_xz,
                  w_others):
    is_found = False
    while not is_found:
        cur_wordlist = get_initial_wordlist8(w_j, w_k, w_q, w_x, w_z, w_w)
        gap_init = constraint_satisfaction_gap_from_wordlist(RARE_TILES, cur_wordlist)
        if random.randint(0, 1) == 1:
            cur_wordlist[:] = cur_wordlist[4:8] + cur_wordlist[0:4]

        # Remove words to resolve excesses of key letters
        for word in cur_wordlist:
            if gap_init['j'] < 0 and 'j' in word:
                cur_wordlist.remove(word)
                gap_init['j'] = 0
                continue
            elif gap_init['k'] < 0 and 'k' in word:
                cur_wordlist.remove(word)
                gap_init['k'] = 0
                continue
            elif gap_init['q'] < 0 and 'q' in word:
                cur_wordlist.remove(word)
                gap_init['q'] = 0
                continue
            elif gap_init['x'] < 0 and 'x' in word:
                cur_wordlist.remove(word)
                gap_init['x'] = 0
                continue
            elif gap_init['z'] < 0 and 'z' in word:
                cur_wordlist.remove(word)
                gap_init['z'] = 0
                continue
            elif gap_init['w'] < 0 and 'w' in word:
                cur_wordlist.remove(word)
                gap_init['w'] = 0
                continue
        gap = constraint_satisfaction_gap_from_wordlist(BANANAGRAM_TILES, cur_wordlist)
        if any(gap_value < 0 for gap_value in gap.values()):
            continue    # Haven't resolved all the excesses yet

        gap_key_count = len([k for k in gap.keys() if k in 'jkqxzw'])
        word_count = len(cur_wordlist)

        if word_count == 8 and gap_key_count == 0:
            return cur_wordlist
        elif word_count == 7:
            if gap_key_count == 0:
                cur_wordlist.append(get_rand_item(w_others))
                return cur_wordlist
            if gap_key_count == 1:
                if gap['j'] == 1:
                    cur_wordlist.append(get_rand_item(w_j))
                    return cur_wordlist
                elif gap['k'] == 1:
                    cur_wordlist.append(get_rand_item(w_k1))
                    return cur_wordlist
                elif gap['q'] == 1:
                    cur_wordlist.append(get_rand_item(w_q))
                    return cur_wordlist
                elif gap['x'] == 1:
                    cur_wordlist.append(get_rand_item(w_x))
                    return cur_wordlist
                elif gap['z'] == 1:
                    cur_wordlist.append(get_rand_item(w_z1))
                    return cur_wordlist
                elif gap['w'] == 1:
                    cur_wordlist.append(get_rand_item(w_w1))
                    return cur_wordlist
                elif gap['k'] == 2:
                    cur_wordlist.append(get_rand_item(w_k2))
                    return cur_wordlist
                elif gap['w'] == 2:
                    cur_wordlist.append(get_rand_item(w_w2))
                    return cur_wordlist
                elif gap['z'] == 2:
                    cur_wordlist.append(get_rand_item(w_z2))
                    return cur_wordlist
            if gap_key_count == 2:
                if gap['j'] == 1 and gap['z'] == 1:
                    cur_wordlist.append(get_rand_item(w_jz))
                    return cur_wordlist
                elif gap['k'] == 1 and gap['q'] == 1:
                    cur_wordlist.append(get_rand_item(w_kq))
                    return cur_wordlist
                elif gap['k'] == 1 and gap['w'] == 1:
                    cur_wordlist.append(get_rand_item(w_kw))
                    return cur_wordlist
                elif gap['q'] == 1 and gap['z'] == 1:
                    cur_wordlist.append(get_rand_item(w_qz))
                    return cur_wordlist
                elif gap['w'] == 1 and gap['z'] == 1:
                    cur_wordlist.append(get_rand_item(w_wz))
                    return cur_wordlist
                elif gap['x'] == 1 and gap['z'] == 1:
                    cur_wordlist.append(get_rand_item(w_xz))
                    return cur_wordlist
        elif word_count < 7:
            if gap_key_count == 1:
                if gap['j'] == 1:
                    cur_wordlist.append(get_rand_item(w_j))
                if gap['k'] == 1:
                    cur_wordlist.append(get_rand_item(w_k1))
                if gap['q'] == 1:
                    cur_wordlist.append(get_rand_item(w_q))
                if gap['x'] == 1:
                    cur_wordlist.append(get_rand_item(w_x))
                if gap['z'] == 1:
                    cur_wordlist.append(get_rand_item(w_z1))
                if gap['w'] == 1:
                    cur_wordlist.append(get_rand_item(w_w1))
                if gap['k'] == 2:
                    cur_wordlist.append(get_rand_item(w_k2))
                if gap['w'] == 2:
                    cur_wordlist.append(get_rand_item(w_w2))
                if gap['z'] == 2:
                    cur_wordlist.append(get_rand_item(w_z2))
            elif gap_key_count >= 2:
                if gap['j'] == 1 and gap['z'] == 1:
                    cur_wordlist.append(get_rand_item(w_jz))
                if gap['k'] == 1 and gap['q'] == 1:
                    cur_wordlist.append(get_rand_item(w_kq))
                if gap['k'] == 1 and gap['w'] == 1:
                    cur_wordlist.append(get_rand_item(w_kw))
                if gap['q'] == 1 and gap['z'] == 1:
                    cur_wordlist.append(get_rand_item(w_qz))
                if gap['x'] == 1 and gap['z'] == 1:
                    cur_wordlist.append(get_rand_item(w_xz))
        if len(cur_wordlist) >= 8:
            return cur_wordlist[:8]


def log_wordlist_and_hash(path_log, words, hash):
    msg_log = '{0:s} {1:s} {2:d}'.format(get_datestamp(), words.__str__(), hash)
    with open(path_log, 'a') as f:
        f.write(msg_log + '\n')


def main(pool, parser_args):
    args_cache = parser_args.cache
    args_lookup = parser_args.lookup
    args_search = parser_args.search

    words12 = read_words12(PATH_WORDS12)
    w_others = [w for w in words12
                if w.count('j') == 0 and w.count('k') == 0 and w.count('q') == 0 and
                w.count('x') == 0 and w.count('z') == 0 and w.count('w') == 0]
    cache4 = WordlistCache()
    if args_lookup > 0:
        wordlist4s = cache4.lookup_hash(args_lookup)
        for wordlist4 in wordlist4s:
            print('{0:s}'.format(wordlist4.__str__()))
    elif args_cache > 0:
        populate_cache(pool, w_others, cache4, args_cache)
    elif args_search:
        search(cache4, words12, w_others)


def populate_cache(pool, w_others, cache4, n):
    chunk_size = int(n / POOL_SIZE)
    async4s = [pool.apply_async(get_valid_wordlist4s, (w_others, chunk_size))
               for proc in ['a', 'b', 'c']]
    wordlist4s = flatten([async4.get(timeout=None) for async4 in async4s])
    for wordlist4 in wordlist4s:
        cache4.add(wordlist4, True)


def print_wordlist(header, words):
    print(header)
    for w in words:
        print('\t{0:s}'.format(w.__str__()))


def read_words12(path):
    with open(path, 'r') as input_file:
        lines = [line.rstrip('\n') for line in input_file]
        w12 = [line for line in lines
               if line.isalpha() and line.islower() and len(line) == WORDLENGTH]
    return w12


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


def search(cache4, words12, w_others):
    cache4.load_keys()
    print('Cache has {0:d} keys'.format(cache4.key_count()))

    w_j = [w for w in words12 if w.count('j') > 0]
    w_k = [w for w in words12 if w.count('k') > 0]
    w_q = [w for w in words12 if w.count('q') > 0]
    w_x = [w for w in words12 if w.count('x') > 0]
    w_z = [w for w in words12 if w.count('z') > 0]
    w_w = [w for w in words12 if w.count('w') > 0]

    w_j1 = [w for w in words12 if w.count('j') == 1]
    w_k1 = [w for w in words12 if w.count('k') == 1]
    w_q1 = [w for w in words12 if w.count('q') == 1]
    w_x1 = [w for w in words12 if w.count('x') == 1]
    w_z1 = [w for w in words12 if w.count('z') == 1]
    w_w1 = [w for w in words12 if w.count('w') == 1]

    w_k2 = [w for w in words12 if w.count('k') == 2]
    w_w2 = [w for w in words12 if w.count('w') == 2]
    w_z2 = [w for w in words12 if w.count('z') == 2]

    w_jz = [w for w in words12 if w.count('j') == 1 and w.count('z') == 1]
    w_kq = [w for w in words12 if w.count('k') == 1 and w.count('q') == 1]
    w_kw = [w for w in words12 if w.count('k') == 1 and w.count('w') == 1]
    w_qz = [w for w in words12 if w.count('q') == 1 and w.count('z') == 1]
    w_wz = [w for w in words12 if w.count('w') == 1 and w.count('z') == 1]
    w_xz = [w for w in words12 if w.count('x') == 1 and w.count('z') == 1]

    cache_lookups = 0
    while True:
        wordlist8 = get_wordlist8(w_j, w_k, w_q, w_x, w_z, w_w,
                                  w_k1, w_z1, w_w1,
                                  w_k2, w_z2, w_w2,
                                  w_jz, w_kq, w_kw, w_qz, w_wz, w_xz,
                                  w_others)
        cntr8 = get_counter_from_wordlist(wordlist8)
        gap4 = constraint_satisfaction_gap_from_counter(BANANAGRAM_TILES, cntr8)
        if any(gap_values < 0 for gap_values in gap4.values()):
            continue
        if cache4.has_hash_of_counter(gap4):
            print_wordlist('\nPossible solution found:', wordlist8)
            hash = cache4.counter2hash(gap4)
            log_wordlist_and_hash(PATH_LOG_SOLUTIONS, wordlist8, hash)
        cache_lookups += 1
        if cache_lookups % 100000 == 0:
            show_progress('K')
        elif cache_lookups % 1000 == 0:
            show_progress('k')


def show_progress(txt):
    print(txt, end='', flush=True)


# Definitions:
#   * Word lists with j's, k's, q's, x's, z's, and w's are called w_j, w_k, etc.
#   * The list called w_others consists solely of words with no j, k, q, x, z, or w.
# Script modes of operation:
#   * CACHE.  Populate a cache with 4-word lists.
#      - Each 4-word list has an integer hash value.
#      - Each 4-word list is stored in a cache files named after the hash value (mod 10K).
#   * LOOKUP.  Look up a particular 4-word list from the cache, by its (integer) hash value.
#   * SEARCH.  Search for 8-word lists that match a complementary 4-word list already in the cache.
#      - First read in the hashes from the cache.  Don't bother reading in the 4-word lists.
#      - If a complementary 4-word list is matched, it can be looked up later, from its hash value.
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='bgrams_12x12')
    parser.add_argument('--cache',
                        default=0,
                        type=int,
                        help='number of 4-word lists to append to hash cache files')
    parser.add_argument('--lookup',
                        default=0,
                        type=int,
                        help='lookup given hash value in hash cache')
    parser.add_argument('--search',
                        # default=0,
                        action='store_true',
                        help='repeatedly search for 8-word lists with a cached complement')

    args = parser.parse_args(sys.argv[1:])
    pool = multiprocessing.Pool(processes=POOL_SIZE)
    try:
        main(pool, args)
    finally:
        pool.close()
        pool.join()
