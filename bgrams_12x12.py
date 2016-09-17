#!/usr/bin/python3

import argparse
import ast
import collections
from functools import reduce
import math
import multiprocessing  # cpu_count, Pool
from operator import mul
from os import listdir
from os.path import isdir, isfile, join
import random
import string
import sys
import time

CACHE_DIR = 'bgrams_12x12_cache'
CACHE_FILE_BASENAME = 'cache_'
CONST_LARGEST_LONG_PRIME = 9223372036854775783  # 2^63 - 25
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
    'u': 6,  'v': 3, 'w': 3,         'y': 3,
})
RARE_TILES = collections.Counter({
    'j': 2, 'k': 2, 'q': 2, 'x': 2, 'z': 2
})
WORDLENGTH = 12


class WordlistCache:
    def __init__(self, cache_dir):
        self._alpha2prime = self._get_alpha2prime_by_frequency()
        self._cache_dir = cache_dir
        if not isdir(self._cache_dir):
            sys.exit('Error: \"{0:s}\" does not exist as a directory'.format(self._cache_dir))
        self._hashset = set()
        self._mod = WordlistCache._get_modulus()

    def add(self, wordlist, do_cache):
        hash = self._wordlist2hash(wordlist)
        self._hashset.add(hash)
        if do_cache:
            path = self._hash2path(hash)
            with open(path, 'a+') as cache_file:
                cache_file.write('{0:d}:{1:s}\n'.format(hash, wordlist.__str__()))

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
                    parts = line.split(':', 1)
                    line_hash = parts[0]
                    self._hashset.add(int(line_hash))
            cache_files_read += 1
            if cache_files_read % 1000 == 0:
                show_progress('C')
            elif cache_files_read % 100 == 0:
                show_progress('c')
        print('')

    def lookup_hash_lines(self, target_hash):
        result = []
        path = self._hash2path(target_hash)
        with open(path, 'r') as cache_file:
            lines = cache_file.readlines()
            for line in lines:
                parts = line.split(':', 1)
                line_hash = parts[0]
                # wordlist = parts[1]
                if int(line_hash) == target_hash:
                    result.append(line)
        return result

    def lookup_hash_wordlists(self, target_hash):
        lines = self.lookup_hash_lines(target_hash)

        def line2wordlist(line):
            parts = line.split(':')
            wordlist_str = parts[1]
            return ast.literal_eval(wordlist_str)
        return map(line2wordlist, lines)

    # primes: 2, 3, 5, ..., 67, 71, 73, 79, 83, 89, 97, 101
    @staticmethod
    def _get_alpha2prime_by_frequency():
        alpha2prime = {
            'e':  2, 's':  3, 'i':  5, 'a':  7, 'r': 11,
            'n': 13, 't': 17, 'o': 19, 'l': 23, 'd': 29,
            'c': 31, 'u': 37, 'g': 41, 'p': 43, 'm': 47,
            'h': 53, 'b': 59, 'y': 61, 'f': 67, 'v': 71,
            'w': 73,
            'j':  0, 'k':  0, 'q':  0, 'x':  0, 'z':  0
        }
        return alpha2prime

    @staticmethod
    def _get_alpha2prime_lexicographic(p_init, p_step):
        alphabet = string.ascii_lowercase[:26]
        primes = WordlistCache._primes_less_than(102)
        alpha2prime = {}
        for i in range(0, 26):
            alpha2prime[alphabet[i]] = primes[p_init + p_step * i]
        return alpha2prime

    # Modulus = One billion supports non-colliding hashes for 9 & 12-letter words,
    # but hash collisions are likely inevitable for the prime multiplication technique with any hash < LONG_MAX.
    @staticmethod
    def _get_modulus():
        # billion = 1000000000  # 10^9
        # trillion = 1000000000000  # 10^12
        # quadrillion = 1000000000000000  # 10^15
        return CONST_LARGEST_LONG_PRIME

    def _hash2path(self, hash):
        return self._cache_dir + '/' + CACHE_FILE_BASENAME + str(hash % 10000).zfill(4)

    @staticmethod
    def _primes_less_than(n):
        aux = {}
        # From O'Reilly's Python Cookbook
        return [aux.setdefault(p, p) for p in range(2, n)
                if 0 not in [p % d for d in aux if p >= d + d]]

    def _wordlist2hash(self, wordlist):
        chars = ''.join(wordlist)
        hash = reduce(mul, [self._alpha2prime[c] for c in chars]) % self._mod
        return hash


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


def get_counter_from_wordlist(words):
    counter = collections.Counter()
    for w in words:
        counter += collections.Counter(w)
    return counter


def get_initial_wordlist8(w_j, w_k, w_q, w_z, w_x):
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

    wordlist = [rand_j0, rand_k0, rand_q0, rand_x0, rand_z0,
                rand_j1, rand_k1, rand_q1, rand_x1, rand_z1
                ]
    # Remove a random selection of 4
    i0 = random.randint(0, 9)
    i1 = random.randint(0, 8)
    del wordlist[i0]
    del wordlist[i1]
    return wordlist  # [:wordcount]


def get_rand_item(words) -> str:
    return random.choice(words)


def get_valid_wordlist4s(w_w1, w_others, num_wordlists):
    valid_wordlist4s = []
    while len(valid_wordlist4s) < num_wordlists:
        wordlist4 = [get_rand_item(w_w1), get_rand_item(w_w1),
                     get_rand_item(w_w1), get_rand_item(w_others)]
        cntr4 = get_counter_from_wordlist(wordlist4)
        gap8 = constraint_satisfaction_gap_from_counter(COMMON_TILES, cntr4)
        if all(gap_values >= 0 for gap_values in gap8.values()):
            valid_wordlist4s.append(wordlist4)
    return valid_wordlist4s


# The code for this function isn't pretty, but it doesn't need to be.
# It returns an 8-word list that ideally contains the right number of
#     j's, k's, q's, x's, z's (2 each), and w's (3)
# The caller checks to see if the wordlist return is actually usable.
def get_wordlist8(w_j, w_k, w_q, w_x, w_z,
                  w_k1, w_z1,
                  w_k2, w_z2,
                  w_jz, w_kq, w_qz, w_wz, w_xz,
                  w_others):
    is_found = False
    while not is_found:
        cur_wordlist = get_initial_wordlist8(w_j, w_k, w_q, w_x, w_z)
        iter_count = 0
        while iter_count < 3:
            iter_count += 1
            gap_init = constraint_satisfaction_gap_from_wordlist(RARE_TILES, cur_wordlist)
            swap_point = random.randint(0, 7)
            if swap_point > 0:
                cur_wordlist[:] = cur_wordlist[swap_point:8] + cur_wordlist[0:swap_point]
            if random.randint(0, 1) == 1:
                cur_wordlist.reverse()

            # Remove words to resolve excesses of rare letters
            for word in cur_wordlist:
                if gap_init['j'] < 0 and 'j' in word:
                    # print('INFO: Wordlist={0:s}'.format(cur_wordlist.__str__()))
                    # print('INFO: Removing excess j word: {0:s}'.format(word))
                    cur_wordlist.remove(word)
                    gap_init['j'] = 0  # Simplification
                    continue
                elif gap_init['k'] < 0 and 'k' in word:
                    # print('INFO: Wordlist={0:s}'.format(cur_wordlist.__str__()))
                    # print('INFO: Removing excess k word: {0:s}'.format(word))
                    cur_wordlist.remove(word)
                    gap_init['k'] = 0  # Simplification
                    continue
                elif gap_init['q'] < 0 and 'q' in word:
                    # print('INFO: Wordlist={0:s}'.format(cur_wordlist.__str__()))
                    # print('INFO: Removing excess q word: {0:s}'.format(word))
                    cur_wordlist.remove(word)
                    gap_init['q'] = 0  # Simplification
                    continue
                elif gap_init['x'] < 0 and 'x' in word:
                    # print('INFO: Wordlist={0:s}'.format(cur_wordlist.__str__()))
                    # print('INFO: Removing excess x word: {0:s}'.format(word))
                    cur_wordlist.remove(word)
                    gap_init['x'] = 0  # Simplification
                    continue
                elif gap_init['z'] < 0 and 'z' in word:
                    # print('INFO: Wordlist={0:s}'.format(cur_wordlist.__str__()))
                    # print('INFO: Removing excess z word: {0:s}'.format(word))
                    cur_wordlist.remove(word)
                    gap_init['z'] = 0  # Simplification
                    continue
            gap = constraint_satisfaction_gap_from_wordlist(BANANAGRAM_TILES, cur_wordlist)
            if any(gap_value < 0 for gap_value in gap.values()):
                break    # Haven't resolved all the excesses yet

            gap_key_count = len([k for k in gap.keys() if k in 'jkqxz'])
            word_count = len(cur_wordlist)

            if word_count == 8 and gap_key_count == 0:
                is_found = True
                break
            if word_count == 7 and gap_key_count == 0:
                cur_wordlist.append(get_rand_item(w_others))
                is_found = True
                break
            if word_count == 8 and gap_key_count == 1:
                if gap['k'] == 1:  # Replace 'k'=>'kk'  or 'q'=>'kq'
                    for i in range(0, len(cur_wordlist)):
                        if cur_wordlist[i].count('k') == 1:
                            cur_wordlist[i] = get_rand_item(w_k2)
                            break
                        elif cur_wordlist[i].count('k') == 0 and cur_wordlist[i].count('q') == 1:
                            cur_wordlist[i] = get_rand_item(w_kq)
                            break
                if gap['x'] == 1:  # Replace 'z'=>'xz'
                    for i in range(0, len(cur_wordlist)):
                        if cur_wordlist[i].count('x') == 0 and cur_wordlist[i].count('z') == 1:
                            cur_wordlist[i] = get_rand_item(w_xz)
                            break
                if gap['z'] == 1:  # Replace 'j'=>'jz' or 'x'=>'xz', or 'z' => 'zz'
                    for i in range(0, len(cur_wordlist)):
                        if cur_wordlist[i].count('z') == 0 and cur_wordlist[i].count('j') == 1:
                            cur_wordlist[i] = get_rand_item(w_jz)
                            break
                        elif cur_wordlist[i].count('z') == 0 and cur_wordlist[i].count('x') == 1:
                            cur_wordlist[i] = get_rand_item(w_xz)
                            break
                        elif cur_wordlist[i].count('z') == 1:
                            cur_wordlist[i] = get_rand_item(w_z2)
                            break
            elif word_count == 7:
                if gap_key_count == 0:
                    cur_wordlist.append(get_rand_item(w_others))
                    is_found = True
                    break
                if gap_key_count >= 2:
                    if gap['j'] == 1 and gap['z'] == 1:
                        cur_wordlist.append(get_rand_item(w_jz))
                    elif gap['k'] == 1 and gap['q'] == 1:
                        cur_wordlist.append(get_rand_item(w_kq))
                    # elif gap['k'] == 1 and gap['w'] == 1:
                    #     cur_wordlist.append(get_rand_item(w_kw))
                    #     return cur_wordlist
                    elif gap['q'] == 1 and gap['z'] == 1:
                        cur_wordlist.append(get_rand_item(w_qz))
                    elif gap['w'] == 1 and gap['z'] == 1:
                        cur_wordlist.append(get_rand_item(w_wz))
                    elif gap['x'] == 1 and gap['z'] == 1:
                        cur_wordlist.append(get_rand_item(w_xz))
                if gap['j'] == 1:
                    cur_wordlist.append(get_rand_item(w_j))
                elif gap['k'] == 1:
                    cur_wordlist.append(get_rand_item(w_k1))
                elif gap['q'] == 1:
                    cur_wordlist.append(get_rand_item(w_q))
                elif gap['x'] == 1:
                    cur_wordlist.append(get_rand_item(w_x))
                elif gap['z'] == 1:
                    cur_wordlist.append(get_rand_item(w_z1))
                elif gap['k'] == 2:
                    cur_wordlist.append(get_rand_item(w_k2))
                elif gap['z'] == 2:
                    cur_wordlist.append(get_rand_item(w_z2))
            elif word_count < 7:
                if gap_key_count >= 2:
                    if gap['j'] == 1 and gap['z'] == 1:
                        cur_wordlist.append(get_rand_item(w_jz))
                    if gap['k'] == 1 and gap['q'] == 1:
                        cur_wordlist.append(get_rand_item(w_kq))
                    if gap['q'] == 1 and gap['z'] == 1:
                        cur_wordlist.append(get_rand_item(w_qz))
                    if gap['x'] == 1 and gap['z'] == 1:
                        cur_wordlist.append(get_rand_item(w_xz))
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
                    if gap['k'] == 2:
                        cur_wordlist.append(get_rand_item(w_k2))
                    if gap['z'] == 2:
                        cur_wordlist.append(get_rand_item(w_z2))

            if len(cur_wordlist) >= 8:
                cur_wordlist = cur_wordlist[:8]
                
            cntr_final = collections.Counter(''.join(cur_wordlist))
            if (len(cur_wordlist) == 8 and cntr_final['j'] == 2 and cntr_final['k'] == 2 and
                        cntr_final['q'] == 2 and cntr_final['x'] == 2 and cntr_final['z'] == 2):
                # gap_final = constraint_satisfaction_gap_from_wordlist(BANANAGRAM_TILES, cur_wordlist)
                # if all(gap_values >= 0 for gap_values in gap_final.values()):
                return cur_wordlist
            else:
                pass  # Continue attempt to modify wordlist
        if not is_found:
            # print('INFO: Passing on iter_count={0:d}: {1:s}'.format(iter_count, cur_wordlist.__str__()))
            pass
    return cur_wordlist


def log_wordlist(path_log, words):
    msg_log = '{0:s};;{1:s};'.format(get_datestamp(), words.__str__())
    with open(path_log, 'a') as f:
        f.write(msg_log + '\n')


def log_wordlist_plus(path_log, hash, words, cache_lookup_count):
    msg_log = '{0:s};{1:d};{2:s};{3:d}'.format(get_datestamp(), hash, words.__str__(), cache_lookup_count)
    with open(path_log, 'a') as f:
        f.write(msg_log + '\n')


def main(pool, parser_args):
    args_cache = parser_args.cache
    args_cache_dir = parser_args.cache_dir
    args_lookup = parser_args.lookup
    args_search = parser_args.search

    words12 = read_words12(PATH_WORDS12)
    cache4 = WordlistCache(args_cache_dir)
    if args_lookup > 0:
        lines = cache4.lookup_hash_lines(args_lookup)
        for line in lines:
            print('{0:s}'.format(line))
    elif args_cache > 0:
        populate_cache(pool, words12, cache4, args_cache)
    elif args_search:
        search(cache4, words12)


# Note: Actual count of items cached is floor(n/POOL_SIZE)
def populate_cache(pool, words12, cache4, n):
    w_commons = [w for w in words12
                 if w.count('j') == 0 and w.count('k') == 0 and w.count('q') == 0 and
                 w.count('x') == 0 and w.count('z') == 0
                 ]
    w_w1 = [w for w in w_commons if w.count('w') == 1]
    w_others = [w for w in w_commons if w.count('w') == 0]
    chunk_size = int(n / POOL_SIZE)
    async4s = [pool.apply_async(get_valid_wordlist4s, (w_w1, w_others, chunk_size))
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


def search(cache4, words12):
    cache4.load_keys()
    print('Cache has {0:d} keys'.format(cache4.key_count()))

    w_w0 = [w for w in words12 if w.count('w') == 0]
    w_j = [w for w in w_w0 if w.count('j') > 0]
    w_k = [w for w in w_w0 if w.count('k') > 0]
    w_q = [w for w in w_w0 if w.count('q') > 0]
    w_x = [w for w in w_w0 if w.count('x') > 0]
    w_z = [w for w in w_w0 if w.count('z') > 0]
    # w_w = [w for w in words12 if w.count('w') > 0]

    w_j1 = [w for w in w_w0 if w.count('j') == 1]
    w_k1 = [w for w in w_w0 if w.count('k') == 1]
    w_q1 = [w for w in w_w0 if w.count('q') == 1]
    w_x1 = [w for w in w_w0 if w.count('x') == 1]
    w_z1 = [w for w in w_w0 if w.count('z') == 1]

    w_k2 = [w for w in w_w0 if w.count('k') == 2]
    w_z2 = [w for w in w_w0 if w.count('z') == 2]

    w_jz = [w for w in w_w0 if w.count('j') == 1 and w.count('z') == 1]
    w_kq = [w for w in w_w0 if w.count('k') == 1 and w.count('q') == 1]
    # w_kw = [w for w in words12 if w.count('k') == 1 and w.count('w') == 1]
    w_qz = [w for w in w_w0 if w.count('q') == 1 and w.count('z') == 1]
    w_wz = [w for w in w_w0 if w.count('w') == 1 and w.count('z') == 1]
    w_xz = [w for w in w_w0 if w.count('x') == 1 and w.count('z') == 1]

    w_others = [w for w in w_w0
                if w.count('j') == 0 and w.count('k') == 0 and w.count('q') == 0 and
                w.count('x') == 0 and w.count('z') == 0]

    cache_lookup_count = 0
    while True:
        wordlist8 = get_wordlist8(w_j, w_k, w_q, w_x, w_z,
                                  w_k1, w_z1,
                                  w_k2, w_z2,
                                  w_jz, w_kq, w_qz, w_wz, w_xz,
                                  w_others)
        # show_progress('.')
        cntr8 = get_counter_from_wordlist(wordlist8)
        gap4 = constraint_satisfaction_gap_from_counter(BANANAGRAM_TILES, cntr8)
        if any(gap_values < 0 for gap_values in gap4.values()):
            continue
        # show_progress('*')
        if cache4.has_hash_of_counter(gap4):
            hash4 = cache4.counter2hash(gap4)
            print_wordlist('\nPossible solution found (hash4={0:d}:'.format(hash4), wordlist8)
            log_wordlist_plus(PATH_LOG_SOLUTIONS, hash4, wordlist8, cache_lookup_count)
            wordlist4s = cache4.lookup_hash_wordlists(hash4)
            for wordlist4 in wordlist4s:
                wordlist12 = wordlist8 + wordlist4
                gap12 = constraint_satisfaction_gap_from_wordlist(BANANAGRAM_TILES, wordlist12)
                if all(gap_values == 0 for gap_values in gap12.values()):
                    print_wordlist('\nCONFIRMED solution found: ', wordlist8)
                    log_wordlist(PATH_LOG_SOLUTIONS, wordlist8)
            cache_lookup_count = 0
        cache_lookup_count += 1
        if cache_lookup_count % 100000 == 0:
            show_progress('K')
        elif cache_lookup_count % 1000 == 0:
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
    parser.add_argument('--cache_dir',
                        default=CACHE_DIR,
                        help='directory where cache is stored')
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
