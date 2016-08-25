#!/usr/bin/python3

# Challenge: Use the tiles from the game Bananagrams to form 7-, 8-, and 9-letter words,
#   and get the highest score possible, given the following:
#     * each 7-letter word is worth 1 point
#     * each 8-letter word is worth 2 points
#     * each 9-letter word is worth 3 points
#     * each unused letter subtracts 1 from your score.
#   The maximum possible score, given the standard 144 tiles, is 48.
#     (From 16 9-letter words, since 16*9=144.)
#   Is it possible to find 16 9-letter words and reach the maximum possible score?
#   If not, what's the highest score that can be achieved?
#
# Strategy:
# (1) Assess how computationally tractable it is to find long lists if 9-letter words with the available tiles.
#       - Start with short candidate lists.  Lengthen as longer solutions are found.
#       - On each iteration:
#         - Remove some of the words that push the character histogram over its constraints.
#         - Add or remove words as needed to get to the desired wordlist length.
#       => Conclusions:
#         - 14-word lists of 9-letter words (that only use available tiles) take a long time to find.
#         - 13-word lists can be found in a few minutes.
#         - 12-word and 11-word lists can be found almost instantaneously.
# (2) Repeatedly find a "stub wordlist" (e.g., an 11-word list), then do an exhaustive search
#       to determine whether or not the remaining tiles can all be used to find the other 9-letter words.
#
# Possible future ideas:
# (*) Be selective about which "stub wordlists" to investigate.
#     - Only pursue those whose gap letter frequencies (i.e., the distribution of the remaining tiles)
#       closely matches (via least squares or cosine distance) the average letter frequencies of 9-letter
#       words in the dictionary.
#     - Exclude from consideration those "stub wordlists"
#         o those with gap letter frequencies that permit too few (e.g., < 100) 9-letter words.
#         o those that permit so many 9-letter words that the number of combinations is too computationally expensive.
#
# TODO: Add command-line arguments, at least to set do_test and is_verbose.
# TODO: Refactor logic for pairs, triples, quadruples, quintuples, to all share the same form.
#
import collections
import math
import multiprocessing  # cpu_count, Pool
import random
import re
import time

PATH_LINUX_WORDS = 'C:\\cygwin64\\usr\\share\\dict\\linux.words'
PATH_LOG_WORDLISTS = 'wordlists.txt'
PATH_WORDS9 = 'C:\\cygwin64\\usr\\share\\dict\\words9'
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
TUPLE_WORDCOUNT = 5

PATH_LOG_WORDLISTS_TEST = 'wordlists_test.txt'
PATH_WORDS9_TEST = 'C:\\cygwin64\\usr\\share\\dict\\words9_test'
TARGET_HISTO_TEST = collections.Counter({
    'a': 12, 'b': 2, 'c': 4, 'd': 5, 'e': 5,
    'f': 0, 'g': 0, 'h': 0, 'i': 5, 'j': 1,
    'k': 1, 'l': 1, 'm': 0, 'n': 1, 'o': 1,
    'p': 0, 'q': 1, 'r': 2, 's': 2, 't': 4,
    'u': 3, 'v': 2, 'w': 0, 'x': 1, 'y': 0,
    'z': 1
})
TARGET_WORDCOUNT_TEST = 6


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


def chunkify(items, number_of_bins):
    chunk_size = math.ceil(len(items) / number_of_bins)
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
    return [item for lst in lsts for item in lst]


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


def get_timestamp():
    return time.strftime('[%Y-%m-%d @ %H:%M:%S]')


def get_tuples2_from_singletons(constraint, singletons1, singletons2):
    pairs = [(w1, w2, c1 + c2)
             for (w1, c1) in singletons1
             for (w2, c2) in singletons2
             if w2 > w1 and is_constraint_satisfied(constraint, c1 + c2)]
    return pairs


def get_tuples3_from_tuples2(constraint, singletons, tuples2):
    pairs = tuples2
    triplets = [(w1, w2, w3, c12 + c3)
                for (w1, w2, c12) in pairs
                for (w3, c3) in singletons
                if w3 > w2 and is_constraint_satisfied(constraint, c12 + c3)]
    return triplets


def get_tuples4_from_tuples3(constraint, singletons, tuples3):
    triples = tuples3
    quadruples = [(w1, w2, w3, w4, c123 + c4)
                  for (w1, w2, w3, c123) in triples
                  for (w4, c4) in singletons
                  if w4 > w3 and is_constraint_satisfied(constraint, c123 + c4)]
    return quadruples


def get_tuples5_from_tuples4(constraint, singletons, tuples4):
    quadruples = tuples4
    quintuples = [(w1, w2, w3, w4, w5)  # Do not return counters; they're no longer needed.
                  for (w1, w2, w3, w4, c1234) in quadruples
                  for (w5, c5) in singletons
                  if w5 > w4 and is_constraint_satisfied(constraint, c1234 + c5)]
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
        f.write(get_timestamp() + ' ' + msg_log + '\n')


def print_wordlist(header, words):
    print(header)
    for w in words:
        print('\t{0:s}'.format(w))


def read_words9(path_words9):
    with open(path_words9) as f:
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


def main(do_test, is_verbose, pool):
    if do_test:
        path_log_wordlists = PATH_LOG_WORDLISTS_TEST
        path_words9 = PATH_WORDS9_TEST
        target_histo = TARGET_HISTO_TEST
        target_wordcount = TARGET_WORDCOUNT_TEST
    else:
        path_log_wordlists = PATH_LOG_WORDLISTS
        path_words9 = PATH_WORDS9
        target_histo = TARGET_HISTO
        target_wordcount = TARGET_WORDCOUNT

    # If there is not a wordlist available with only 9-letter words,
    # then call write_words9() to create one.
    words9 = read_words9(path_words9)
    w_j = get_words_with_char(words9, 'j')
    w_k = get_words_with_char(words9, 'k')
    w_q = get_words_with_char(words9, 'q')
    w_x = get_words_with_char(words9, 'x')
    w_z = get_words_with_char(words9, 'z')
    if is_verbose:
        print("words9 count = {0:d}".format(len(words9)))
        print("words_j count = {0:d}".format(len(w_j)))
        print("words_k count = {0:d}".format(len(w_k)))
        print("words_q count = {0:d}".format(len(w_q)))
        print("words_x count = {0:d}".format(len(w_x)))
        print("words_z count = {0:d}".format(len(w_z)))

    wordlist0 = get_initial_list(words9, w_j, w_k, w_q, w_x, w_z,
                                 wordcount=target_wordcount - TUPLE_WORDCOUNT)
    if is_verbose:
        print_wordlist('Initial candidate word list:', wordlist0)

    cur_wordlist = wordlist0
    iter_count = 0
    is_full_wordlist_found = False
    while not is_full_wordlist_found:
        iter_count += 1
        constraint = constraint_satisfaction_gap_from_wordlist(target_histo, cur_wordlist)
        if constraint is not None:
            if is_verbose:
                print('cur_wordlist={0:s}; gap={1:s}'.format(cur_wordlist.__str__(), constraint.__str__()))
            show_progress('\n' + get_timestamp() + u' \u00bf')  # Upside-down question mark
            singletons = get_singletons(constraint, words9)
            show_progress('{0}'.format(len(singletons)))

            singleton_chunks = chunkify(singletons, multiprocessing.cpu_count())  # Alternative: # CPUs - 1

            # pairs = get_tuples2_from_singletons(constraint, singletons)
            pair_chunk_asyncs = [pool.apply_async(get_tuples2_from_singletons, (constraint, singleton_chunk, singletons))
                                 for singleton_chunk in singleton_chunks]
            pair_chunks = [pair_chunk.get(timeout=None) for pair_chunk in pair_chunk_asyncs]
            pairs = flatten(pair_chunks)
            show_progress(',{0}'.format(len(pairs)))

            # triples = get_tuples3_from_tuples2(constraint, singletons, pairs)
            triple_chunk_asyncs = [pool.apply_async(get_tuples3_from_tuples2, (constraint, singleton_chunk, pairs))
                                   for singleton_chunk in singleton_chunks]
            triple_chunks = [triple_chunk.get(timeout=None) for triple_chunk in triple_chunk_asyncs]
            triples = flatten(triple_chunks)
            show_progress(',{0}'.format(len(triples)))

            # quadruples = get_tuples4_from_tuples3(constraint, singletons, triples)
            quadruple_chunk_asyncs = [pool.apply_async(get_tuples4_from_tuples3, (constraint, singleton_chunk, triples))
                                      for singleton_chunk in singleton_chunks]
            quadruples = flatten([quadruple_chunk.get(timeout=None) for quadruple_chunk in quadruple_chunk_asyncs])
            show_progress(',{0}'.format(len(quadruples)))

            # quintuples = get_tuples5_from_tuples4(constraint, singletons, quadruples)
            quintuple_chunk_asyncs = [
                pool.apply_async(get_tuples5_from_tuples4, (constraint, singleton_chunk, quadruples))
                for singleton_chunk in singleton_chunks]
            quintuples = flatten([quintuple_chunk.get(timeout=None) for quintuple_chunk in quintuple_chunk_asyncs])
            show_progress(',{0}'.format(len(quintuples)))

            show_progress('?')

            if len(quintuples) > 0:
                for (w1, w2, w3, w4, w5) in quintuples:
                    full_list = cur_wordlist + [w1, w2, w3, w4, w5]
                    log_wordlist(path_log_wordlists, full_list.__str__())
                    print_wordlist('\nFull {0:d}-word wordlist found!'.format(target_wordcount), full_list)
                is_full_wordlist_found = True
            else:
                # Start over again
                cur_wordlist = get_initial_list(words9, w_j, w_k, w_q, w_x, w_z,
                                                wordcount=target_wordcount - TUPLE_WORDCOUNT)
        if not is_full_wordlist_found:
            # if iter_count % 1000 == 0:
            show_progress('.')
            if constraint is None:  # Word-list invalid.  Try again.
                excess_letters = [k for k, v in get_gap_histo(cur_wordlist).items() if v < 0]
                if len(excess_letters) > 0:
                    remove_excess_letters(excess_letters, cur_wordlist)

                missing_letters = [k for k, v in get_gap_histo(cur_wordlist).items() if v > 0]
                if len(missing_letters) > 0:
                    add_missing_letters(w_j, w_k, w_q, w_x, w_z, missing_letters, cur_wordlist)
                adjust_word_count(words9, target_wordcount - TUPLE_WORDCOUNT, cur_wordlist)


if __name__ == '__main__':
    # Parse command-line args
    do_test = False
    is_verbose = False
    pool = multiprocessing.Pool(processes=4)
    try:
        main(do_test, is_verbose, pool)
    finally:
        pool.close()
        pool.join()
