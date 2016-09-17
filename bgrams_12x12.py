#!/usr/bin/python3

import argparse
import ast
from ast import literal_eval
from collections import Counter, defaultdict
from functools import reduce
from multiprocessing import Pool
from operator import mul
from os import listdir, makedirs
from os.path import isdir, isfile, join
import sys
from time import strftime

BASENAME_CACHE_FILE = 'cache_'

CONST_LARGEST_LONG_PRIME = 9223372036854775783  # 2^63 - 25
CONST_LARGEST_PRIME_UNDER_TEN_K = 9973
CONST_ONE_MILLION = 1000000

PRIMES_26 = [101, 97, 89, 83, 79, 73, 71, 67, 61, 59,
             53, 47, 43, 41, 37, 31, 29, 23, 19, 17,
             13, 11, 7, 5, 3, 2]

# See Wikipedia page on Bananagrams
TILES_COUNTER = Counter({
    'a': 13, 'b': 3, 'c': 3, 'd':  6, 'e': 18,
    'f':  3, 'g': 4, 'h': 3, 'i': 12, 'j':  2,
    'k':  2, 'l': 5, 'm': 3, 'n':  8, 'o': 11,
    'p':  3, 'q': 2, 'r': 9, 's':  6, 't':  9,
    'u':  6, 'v': 3, 'w': 3, 'x':  2, 'y':  3,
    'z':  2
})

TILES_RARE = 'jqxz'  # q j x z
TILES_UNCOMMON = 'bfkvwy'  # w v k f y b
TILES_COMMON = 'hmpgucdlotnraise'  # All others (acdeghilmnoprstu), in ascending order of freq

TILES_ALPHALIST = list(map(lambda c: TILES_COUNTER[c], TILES_RARE + TILES_UNCOMMON + TILES_COMMON))
TILES_UNCOMMON_HEXLIST = list(map(lambda c: TILES_COUNTER[c], TILES_UNCOMMON))

# PATH_DICT = '/usr/share/dict/linux.words'
PATH_ALPHALIST_CACHE_DIR = './__cache26__'
PATH_DECALIST_CACHE_DIR = './__cache10__'
PATH_HEXLIST_CACHE_DIR = './__cache06__'
PATH_WORDLIST_CACHE_DIR = './__wordlistcache__'

PATH_LOG_FILE = './bgrams_12x12_phases.log'
PATH_WORDS12 = './bgrams12_test.txt'  # './bgrams12_test.txt'
POOL_SIZE = 3


class MultiDict:
    """
    A dict in which the values are lists.
    """
    def __init__(self, keylen):
        if keylen < 10 or keylen > 26:
            raise Exception('Error: unknown key length: {0:d}'.format(keylen))
        self._dict = defaultdict(list)
        self._keylen = keylen

    def __getitem__(self, key):
        return self._dict[self.key2hash(key, self._keylen)]

    def __setitem__(self, key, val):
        self.add(key, val)

    def add(self, key, val):
        h = MultiDict.key2hash(key, self._keylen)
        self._dict[h].append(val)

    def itemcount(self):
        return sum([len(val) for val in self._dict.values()])

    @staticmethod
    def key2hash(key, keylen):
        primes = PRIMES_26
        return reduce(mul, [p * n for (p, n) in zip(primes[-keylen:], key) if n > 0], 1)

    def print(self, header):
        max_display_len = 1
        print('{0:s} (#k={1:d}, #v={2:d}):'.format(
            header, len(list(self._dict.keys())), self.itemcount()
            ))
        for key in self._dict.keys():
            val = self._dict[key]
            ellipsis = ' ... ({0:d})'.format(len(val))
            print('  key={0:s}, val={1:s}{2:s}'.format(
                key,
                val[0:max_display_len].__str__(),
                ellipsis if len(val) > max_display_len else ''
                ))

    def test_sample_value(self):
        return list(self._dict.values())[0]


class PolyListCache:
    def __init__(self, path_dir, intlist_len):
        self._path_dir = path_dir
        self._intlist_len = intlist_len

    def cache(self, polylists, h):
        # if not all(len(x) == self._intlist_len for x in polylists):  # TODO: Performance impact for this check?
        #     raise Exception('polylist of len len={0:d} being added to cache for items of len={1:d}'.format(
        #         len(polylists[0]), self._intlist_len
        #     ))
        msg = polylists.__str__()
        path_file = self._path_dir + '/' + BASENAME_CACHE_FILE + str(h % 1000).zfill(4)
        with open(path_file, 'a') as f:
            f.write(msg + '\n')

    def get_all(self):
        for rel_cache_file in listdir(self._path_dir):
            path_file = join(self._path_dir, rel_cache_file)
            with open(path_file, 'r') as cache_file:
                lines = cache_file.readlines()
                for line in lines:
                    polylists = ast.literal_eval(line)
                    # if not all(len(x) == self._intlist_len for x in polylists):  # TODO: Performance impact for this check?
                    #     raise Exception('polylist of len={0:d} found in PolylistCache for items of len={1:d}: {2:s}'.format(
                    #         len(polylists[0]), self._intlist_len, polylists.__str__()
                    #     ))
                    yield polylists


class PolyListHashCache:  # PolyListHashCache might be a more accurate term.
    """A cache (on-disk and in-memory) of lists of six ints, representing letter counts
      * On disk, both the hashes and values are stored.
      * In memory, only the hashes are used.
    If a memory check finds a cached hash key, the corresponding value can be looked up from disk.
      * keys: ints (from 0 to LONG_MAX)
      * values: lists of lists of ints.

    TODO: Validate the assumption that the hash in use is the same one that was used to build the hash.
      * Implementation: Store a file (e.g., .hash_test) with test data in the cache, together with expected hashes.
    """
    def __init__(self, cache_dir, intlist_len):
        self._cache_dir = cache_dir
        self._file_basename = 'cache_'
        self._intlist_len = intlist_len
        self._primes = PRIMES_26
        if not isdir(self._cache_dir):
            sys.exit('Error: \"{0:s}\" does not exist as a directory'.format(self._cache_dir))
        self._hashset = set()
        self._mod = PolyListHashCache._get_modulus()

    def add(self, do_log, do_mem, value):
        assert(len(value) == self._intlist_len)
        h = self.value2hash(value)
        if do_log:
            path_cache_file = self._hash2path(h)
            with open(path_cache_file, 'a+') as cache_file:
                cache_file.write('{0:d}:{1:s}\n'.format(h, value.__str__()))
        if do_mem:
            self._hashset.add(h)

    def has_hash(self, h):
        return h in self._hashset

    def has_hash_of_value(self, value):
        return self.has_hash(self.value2hash(value))

    def key_count(self):
        return len(self._hashset)

    def load_keys(self):
        cache_paths = [join(self._cache_dir, file) for file in listdir(self._cache_dir)
                       if isfile(join(self._cache_dir, file))]
        print('Loading cache: ', end='')
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
        if isfile(path):
            with open(path, 'r') as cache_file:
                lines = cache_file.readlines()
                for line in lines:
                    parts = line.split(':', 1)
                    line_hash = parts[0]
                    # value = parts[1]
                    if int(line_hash) == target_hash:
                        result.append(line)
        return result

    def lookup_hash_values(self, target_hash):
        lines = self.lookup_hash_lines(target_hash)

        def line2value(line):
            parts = line.split(':')
            value_str = parts[1]
            return literal_eval(value_str)
        return map(line2value, lines)

    def value2hash(self, value):
        h = 1
        for intlist in value:
            h *= reduce(mul, [p * x for (p, x) in zip(self._primes[-self._intlist_len:], intlist) if x > 0], 1)
        return h % self._mod

    @staticmethod
    def _get_modulus():
        return CONST_LARGEST_LONG_PRIME  # 2^63 - 25

    def _hash2path(self, h):
        return self._cache_dir + '/' + BASENAME_CACHE_FILE + str(h % 10000).zfill(4)


def cache_wordlist(path_log_dir, wordlist):
    msg_log = wordlist.__str__()
    path_log_file = path_log_dir + '/wordlists'
    with open(path_log_file, 'a+') as f:
        f.write(msg_log + '\n')


def counter2polylist(cntr, intlist_len):
    if intlist_len < 10 or intlist_len > 26:
        raise Exception('counter2polylist: intlist_len ({0:d}) is out of range'.format(intlist_len))
    letters = TILES_RARE + TILES_UNCOMMON + TILES_COMMON
    return [cntr[c] for c in letters[:intlist_len]]


def get_cache_dir(dir_type, batch):
    if type(dir_type).__name__ == 'str':
        if dir_type == 'alpha':
            dir_name = PATH_ALPHALIST_CACHE_DIR + '/' + batch
        elif dir_type == 'deca':
            dir_name = PATH_DECALIST_CACHE_DIR + '/' + batch
        # elif dir_type == 'hex':
        #     dir_name = PATH_HEXLIST_CACHE_DIR
        elif dir_type == 'word':
            dir_name = PATH_WORDLIST_CACHE_DIR
        else:
            raise 'Error: Unknown cache directory type: {0:s}'.format(dir_type)
    elif type(dir_type).__name__ == 'int':
        dir_name = '__cache{0:s}__/{1:s}'.format(str(dir_type).zfill(2), batch)
    else:
        raise Exception('Unknown cache directory specifier: {0:s}'.format(dir_type.__str__()))
    return dir_name


def get_datestamp():
    return strftime('[%Y-%m-%d @ %H:%M:%S]')


def get_decalists_from_words(words12):
    decalists = []
    decalist_hashes_seen = set()
    primes10 = [29, 23, 19, 17, 13, 11, 7, 5, 3, 2]
    for w in words12:
        cntr = Counter(w)
        decalist = [cntr.get(c, 0) for c in TILES_RARE + TILES_UNCOMMON]
        h = reduce(mul, [pow(p, x) for (p, x) in zip(primes10, decalist) if x > 0], 1)
        if h in decalist_hashes_seen:
            pass
        else:
            decalists.append(decalist)
            decalist_hashes_seen.add(h)
    return decalists


def get_hexlists(decalists, rare_counts):
    (arg_j, arg_q, arg_x, arg_z) = rare_counts
    filtered = [
        [b, f, k, v, w, y] for [j, q, x, z, b, f, k, v, w, y] in decalists
        if j == arg_j == j and q == arg_q and x == arg_x and arg_z == z
        ]
    return filtered


def is_gap_valid(gap):
    return all(x >= 0 for x in gap)


def is_gap_zero(gap):
    return all(x == 0 for x in gap)


def list_add(list0, list1):
    diff = [item0 + item1 for (item0, item1) in zip(list0, list1)]
    return diff


def list_sub(list0, list1):
    diff = [item0 - item1 for (item0, item1) in zip(list0, list1)]
    return diff


def list_subs(list0, lists):
    total = list_sum(lists)
    diff = list_sub(list0, total)
    return diff


def list_sum(lists):
    total = reduce(list_add, lists)
    return total


def log_msg(msg):
    if msg == '':
        msg_plus = ''
    else:
        msg_plus = '{0:s} {1:s}'.format(get_datestamp(), msg)
    with open(PATH_LOG_FILE, 'a+') as log_file:
        log_file.write('{0:s}\n'.format(msg_plus))
    print(msg_plus)


def mkdirs():
    makedirs(PATH_ALPHALIST_CACHE_DIR, exist_ok=True)
    makedirs(PATH_DECALIST_CACHE_DIR, exist_ok=True)
    makedirs(PATH_HEXLIST_CACHE_DIR, exist_ok=True)
    makedirs(PATH_WORDLIST_CACHE_DIR, exist_ok=True)
    for batch in ['alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot',
                  'golf', 'hotel', 'india', 'juliet', 'kilo']:
        for segment in range(10, 27):
            makedirs(get_cache_dir(segment, batch), exist_ok=True)


def phase1(cache4, decalists, batch):
    rares = {}
    hexlists = dict()

    rares['a'] = [1, 0, 0, 0]
    hexlists['a'] = get_hexlists(decalists, [1, 0, 0, 0])

    rares['b'] = [1, 0, 0, 1]
    hexlists['b'] = get_hexlists(decalists, [1, 0, 0, 1])

    rares['c'] = [0, 1, 0, 0]
    hexlists['c'] = get_hexlists(decalists, [0, 1, 0, 0])

    rares['d'] = [0, 1, 0, 1]
    hexlists['d'] = get_hexlists(decalists, [0, 1, 0, 1])

    rares['e'] = [0, 0, 1, 0]
    hexlists['e'] = get_hexlists(decalists, [0, 0, 1, 0])

    rares['f'] = [0, 0, 1, 1]
    hexlists['f'] = get_hexlists(decalists, [0, 0, 1, 1])

    rares['g'] = [0, 0, 0, 2]
    hexlists['g'] = get_hexlists(decalists, [0, 0, 0, 2])

    rares['h'] = [0, 0, 0, 1]
    hexlists['h'] = get_hexlists(decalists, [0, 0, 0, 1])

    rares['x'] = [0, 0, 0, 0]
    hexlists['x'] = get_hexlists(decalists, [0, 0, 0, 0])

    def str2hexlists(s):
        result = list(map(lambda c: hexlists[c], s))
        return result

    def str2rares(s):
        result = list(map(lambda c: rares[c], s))
        return result

    if batch == 'alpha':
        srcs_str = 'aacceegx'
        phase1_batch(
            cache4, 'alpha',
            rares=str2rares(srcs_str),
            srcs=str2hexlists(srcs_str),
            do_ltcheck_10=True, do_ltcheck_32=True, do_ltcheck_54=True, do_ltcheck_76=False
        )
    elif batch == 'bravo':
        srcs_str = 'aacceehh'
        phase1_batch(
            cache4, 'bravo',
            rares=str2rares(srcs_str),
            srcs=str2hexlists(srcs_str),
            do_ltcheck_10=True, do_ltcheck_32=True, do_ltcheck_54=True, do_ltcheck_76=True
        )
    elif batch == 'charlie':
        srcs_str = 'aaccefgx'
        phase1_batch(
            cache4, 'charlie',
            rares=str2rares(srcs_str),
            srcs=str2hexlists(srcs_str),
            do_ltcheck_10=True, do_ltcheck_32=True, do_ltcheck_54=False, do_ltcheck_76=False
        )
    elif batch == 'delta':
        srcs_str = 'aaccffxx'
        phase1_batch(
            cache4, 'delta',
            rares=str2rares(srcs_str),
            srcs=str2hexlists(srcs_str),
            do_ltcheck_10=True, do_ltcheck_32=True, do_ltcheck_54=True, do_ltcheck_76=True
        )
    elif batch == 'echo':
        srcs_str = 'aacdeegx'
        phase1_batch(
            cache4, 'echo',
            rares=str2rares(srcs_str),
            srcs=str2hexlists(srcs_str),
            do_ltcheck_10=True, do_ltcheck_32=False, do_ltcheck_54=True, do_ltcheck_76=False
        )
    elif batch == 'foxtrot':
        srcs_str = 'aacdefxx'
        phase1_batch(
            cache4, 'foxtrot',
            rares=str2rares(srcs_str),
            srcs=str2hexlists(srcs_str),
            do_ltcheck_10=True, do_ltcheck_32=False, do_ltcheck_54=False, do_ltcheck_76=True
        )
    elif batch == 'golf':
        srcs_str = 'aaddeexx'
        phase1_batch(
            cache4, 'golf',
            rares=str2rares(srcs_str),
            srcs=str2hexlists(srcs_str),
            do_ltcheck_10=True, do_ltcheck_32=True, do_ltcheck_54=True, do_ltcheck_76=True
        )
    elif batch == 'hotel':
        srcs_str = 'abcceegx'
        phase1_batch(
            cache4, 'hotel',
            rares=str2rares(srcs_str),
            srcs=str2hexlists(srcs_str),
            do_ltcheck_10=False, do_ltcheck_32=True, do_ltcheck_54=True, do_ltcheck_76=False
        )
    elif batch == 'india':
        srcs_str = 'abccefxx'
        phase1_batch(
            cache4, 'india',
            rares=str2rares(srcs_str),
            srcs=str2hexlists(srcs_str),
            do_ltcheck_10=False, do_ltcheck_32=True, do_ltcheck_54=False, do_ltcheck_76=True
        )
    elif batch == 'juliet':
        srcs_str = 'abcdeexx'
        phase1_batch(
            cache4, 'juliet',
            rares=str2rares(srcs_str),
            srcs=str2hexlists(srcs_str),
            do_ltcheck_10=False, do_ltcheck_32=False, do_ltcheck_54=True, do_ltcheck_76=True
        )
    elif batch == 'kilo':
        srcs_str = 'bbcceexx'
        phase1_batch(
            cache4, 'kilo',
            rares=str2rares(srcs_str),
            srcs=str2hexlists(srcs_str),
            do_ltcheck_10=True, do_ltcheck_32=True, do_ltcheck_54=True, do_ltcheck_76=True
        )
    else:
        raise Exception('Unknown batch: {0:s}'.format(batch))


def phase1_batch(cache4, batch, rares, srcs, do_ltcheck_10, do_ltcheck_32, do_ltcheck_54, do_ltcheck_76):
    """Search for and log decalists into the decalist cache directory, PATH_DECALIST_CACHE_DIR."""
    log_msg('Entering phase 1 for batch {0:s}'.format(batch))
    gap12 = TILES_UNCOMMON_HEXLIST
    polycache10 = PolyListCache(get_cache_dir('deca', batch), 10)
    cache4_lookup_count = 0
    for s0 in srcs[0]:
        for s1 in srcs[1]:
            if do_ltcheck_10 and s1 < s0:
                continue
            show_progress('1')
            for s2 in srcs[2]:
                for s3 in srcs[3]:
                    if do_ltcheck_32 and s3 < s2:
                        continue
                    gap8 = list_subs(gap12, [s0, s1, s2, s3])
                    if not is_gap_valid(gap8):
                        continue
                    show_progress('3')
                    for s4 in srcs[4]:
                        for s5 in srcs[5]:
                            if do_ltcheck_54 and s5 < s4:
                                continue
                            gap6 = list_subs(gap8, [s4, s5])
                            if not is_gap_valid(gap6):
                                continue
                            # show_progress('5')
                            for s6 in srcs[6]:
                                gap5 = list_sub(gap6, s6)
                                if not is_gap_valid(gap5):
                                    continue
                                for s7 in srcs[7]:
                                    if do_ltcheck_76 and s7 < s6:
                                        continue
                                    # show_progress('7')
                                    cache4_lookup_count += 1
                                    gap4 = list_sub(gap5, s7)
                                    hash4 = cache4.value2hash([gap4])
                                    complementary4s = cache4.lookup_hash_values(hash4)
                                    for complementary4 in complementary4s:
                                        gap0 = list_subs(gap4, complementary4)
                                        if all(x == 0 for x in gap0):
                                            ss = [s0, s1, s2, s3, s4, s5, s6, s7]
                                            decalist_left8 = [rare + hexlist for (rare, hexlist) in zip(rares, ss)]
                                            decalist_right4 = [[0, 0, 0, 0] + comp for comp in complementary4]
                                            decalist12 = decalist_left8 + decalist_right4
                                            polycache10.cache(decalist12, hash4)
    log_msg('Exiting phase 1 for batch {0:s}.  Cache lookup count={1:d}'.format(batch, cache4_lookup_count))


def phase2(words12, batch):
    phase2_segment(words12, batch, 10, 26)
    # for len1 in range(10, 26):
    #     phase2_segment(words12, batch, len1, len1 + 1)


def phase2_segment(words12, batch, len1, len2):
    log_msg('Entering phase 2 ({0:d}, {1:d}) for batch {2:s}'.format(len1, len2, batch))
    map_len1_to_len2s = MultiDict(keylen=len1)
    for w in words12:
        cntr = Counter(w)
        polylist_key = counter2polylist(cntr, len1)
        polylist_val = counter2polylist(cntr, len2)
        map_len1_to_len2s[polylist_key] = polylist_val

    gap12 = TILES_ALPHALIST[0:len2]

    # TODO: Look at itertools.product for ideas on how to improve this loop
    polycache_len1 = PolyListCache(get_cache_dir(len1, batch), len1)
    polycache_len2 = PolyListCache(get_cache_dir(len2, batch), len2)
    polylist_len1_count = 0
    for polylists_len1 in polycache_len1.get_all():
        polylist_len1_count += 1
        if polylist_len1_count % CONST_ONE_MILLION == 0:
            show_progress('M')
        elif polylist_len1_count % 1000 == 0:
            show_progress('k')
        for a0 in map_len1_to_len2s[polylists_len1[0]]:
            for a1 in map_len1_to_len2s[polylists_len1[1]]:
                for a2 in map_len1_to_len2s[polylists_len1[2]]:
                    for a3 in map_len1_to_len2s[polylists_len1[3]]:
                        gap8 = list_subs(gap12, [a0, a1, a2, a3])
                        if not is_gap_valid(gap8):
                            continue
                        for a4 in map_len1_to_len2s[polylists_len1[4]]:
                            for a5 in map_len1_to_len2s[polylists_len1[5]]:
                                gap6 = list_subs(gap8, [a4, a5])
                                if not is_gap_valid(gap6):
                                    continue
                                for a6 in map_len1_to_len2s[polylists_len1[6]]:
                                    for a7 in map_len1_to_len2s[polylists_len1[7]]:
                                        gap4 = list_subs(gap6, [a6, a7])
                                        if not is_gap_valid(gap4):
                                            continue
                                        for a8 in map_len1_to_len2s[polylists_len1[8]]:
                                            for a9 in map_len1_to_len2s[polylists_len1[9]]:
                                                gap2 = list_subs(gap4, [a8, a9])
                                                if not is_gap_valid(gap2):
                                                    continue
                                                for a10 in map_len1_to_len2s[polylists_len1[10]]:
                                                    gap1 = list_sub(gap2, a10)
                                                    if not is_gap_valid(gap1):
                                                        continue
                                                    for a11 in map_len1_to_len2s[polylists_len1[11]]:
                                                        gap0 = list_sub(gap1, a11)
                                                        if is_gap_valid(gap0):
                                                            polylist_len2s = [a0, a1, a2, a3, a4, a5,
                                                                              a6, a7, a8, a9, a10, a11]
                                                            h = MultiDict.key2hash(list_add(a0, a11), len2)
                                                            polycache_len2.cache(polylist_len2s, h)
    log_msg('Exiting phase 2 ({0:d}, {1:d}) for batch {2:s}'.format(len1, len2, batch))


def phase3(words12, batch):
    log_msg('Entering phase 3 for batch {0:s}'.format(batch))
    alphalist2words = MultiDict(keylen=26)
    for w in words12:
        cntr = Counter(w)
        alphalist = counter2polylist(cntr, 26)
        alphalist2words[alphalist] = w

    # Note: Most of the values of alphalist2words are singletons.
    # Values of alphalist2words that aren't singletons correspond to
    #   anagrams of 12-letter words, such as antimosquito and misquotation.
    polycache26 = PolyListCache(get_cache_dir(26, batch), 26)
    for alphalists in polycache26.get_all():
        for w0 in alphalist2words[alphalists[0]]:
            for w1 in alphalist2words[alphalists[1]]:
                for w2 in alphalist2words[alphalists[2]]:
                    for w3 in alphalist2words[alphalists[3]]:
                        for w4 in alphalist2words[alphalists[4]]:
                            for w5 in alphalist2words[alphalists[5]]:
                                for w6 in alphalist2words[alphalists[6]]:
                                    for w7 in alphalist2words[alphalists[7]]:
                                        for w8 in alphalist2words[alphalists[8]]:
                                            for w9 in alphalist2words[alphalists[9]]:
                                                for w10 in alphalist2words[alphalists[10]]:
                                                    for w11 in alphalist2words[alphalists[11]]:
                                                        path_wordlist_dir = get_cache_dir('word', batch)
                                                        wordlist12 = [w0, w1, w2, w3, w4, w5,
                                                                      w6, w7, w8, w9, w10, w11]
                                                        cache_wordlist(path_wordlist_dir, wordlist12)
    log_msg('Exiting phase 3 for batch {0:s}'.format(batch))


def populate_hexlistcache(cache4, decalists):
    hexlists_x = get_hexlists(decalists, (0, 0, 0, 0))
    for h0 in hexlists_x:
        show_progress('0')
        for h1 in hexlists_x:
            if h1 < h0:
                continue
            show_progress('1')
            for h2 in hexlists_x:
                if h2 < h1:
                    continue
                for h3 in hexlists_x:
                    if h3 < h2:
                        continue
                    hs = [h0, h1, h2, h3]
                    cntr4 = list_sum(hs)
                    gap8 = list_sub(TILES_UNCOMMON_HEXLIST, cntr4)
                    if is_gap_valid(gap8):
                        cache4.add(True, False, hs)


def read_words12(path):
    with open(path, 'r') as input_file:
        lines = [line.rstrip('\n') for line in input_file]
        w12 = [line for line in lines
               if line.isalpha() and line.islower()]
    return w12


def report_wordlists():
    def get_wordlists_from_cache():
        path_wordlist_dir = get_cache_dir('word', None)
        for rel_cache_file in listdir(path_wordlist_dir):
            path_cache_file = join(path_wordlist_dir, rel_cache_file)
            with open(path_cache_file, 'r') as cache_file:
                wordlists = cache_file.readlines()
                for wordlist in wordlists:
                    yield wordlist
    for wordlist in get_wordlists_from_cache():
        print(wordlist.__str__())


def show_progress(txt):
    print(txt, end='', flush=True)


def main(pool, args):
    if args.phase < -1 or args.phase > 3:
        raise Exception('Error: Illegal value given for phase')  # TODO: Have parser handle input errors
    words12 = read_words12(PATH_WORDS12)
    decalists = get_decalists_from_words(words12)

    if args.mkdirs:
        mkdirs()

    cache4 = PolyListHashCache(PATH_HEXLIST_CACHE_DIR, intlist_len=4)

    if args.cache:
        populate_hexlistcache(cache4, decalists)
    # if args.phase >= 0:
    #   validate_cache(cache4)
    if args.phase == 0:
        phase1(cache4, decalists, args.batch)
        phase2(words12, args.batch)
        phase3(words12, args.batch)
    elif args.phase == 1:
        phase1(cache4, decalists, args.batch)
    elif args.phase == 2:
        phase2(words12, args.batch)
    if args.phase == 3:
        phase3(words12, args.batch)
    if args.report:
        report_wordlists()
    log_msg('')

if __name__ == '__main__':
    """Short guide: Run with --mkdirs, then --cache, then --phase 0 --batch alpha, etc., and finally with --report"""
    # TODO: Exclude improper argument combinations.
    # TODO: Ensure there aren't any repated wordlists in the final report

    parser = argparse.ArgumentParser(
        #'Search for a dozen 12-letter words made out of the 144 Bananagram tiles',
        prog='bgrams_12x12_phases.py',
        formatter_class=argparse.RawTextHelpFormatter
    )
    intent = parser.add_mutually_exclusive_group()
    intent.add_argument('--mkdirs',
                        action='store_true',
                        help='create the cache directories used by this program')
    intent.add_argument('--cache',
                        action='store_true',
                        help='create a cache used in phase 1\n' +
                             '  This "4x6" cache contains 4-tuples of 6-digit signatures, called hexlists'
                        )
    # TODO: Restrict user-supplied values for phase to 0..3
    intent.add_argument('--phase',
                        type=int,
                        choices=range(0, 4),
                        default=-1,
                        help='0=>All (combines phases 1, 2, and 3, below)\n' +
                             'With --phase, the --batch option is required.\n' +
                             'For each batch, phases must be run in order: 1, then 2, then 3'
                             '  1=>Find 10-char signatures\n' +
                             '  2=>Find 24-char signatures\n' +
                             '  3=>Find wordlists'
                        )
    parser.add_argument('--batch',
                        help='for phase 1 or 2: alpha, bravo, charlie, delta, echo,\n' +
                             '\tfoxtrot, golf, hotel, india, juliet, or kilo')
    intent.add_argument('--report',
                        action='store_true',
                        help='display wordlists found')
    args = parser.parse_args(sys.argv[1:])
    if args.batch and (args.mkdirs or args.cache or args.report):
        print('Error: The --batch option should only be used with --phase option.')
        parser.print_help()
        sys.exit(1)

    pool = Pool(processes=POOL_SIZE)
    try:
        main(pool, args)
    finally:
        pool.close()
        pool.join()
