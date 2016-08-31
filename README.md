# Bananagrams
Search for a way to group the tiles into 9-letter words.

**Note:** The script in this directory does **not** attempt to play the game Bananagrams.

**Short version:** Find 16 nine-letter words using the tiles from a game of Bananagrams.

**Details:**

* **Challenge:** Use the tiles from the game Bananagrams to form 7-, 8-, and 9-letter words,
  and get the highest score possible, given the following:
  * each 7-letter word is worth 1 point
  * each 8-letter word is worth 2 points
  * each 9-letter word is worth 3 points
  * each unused letter subtracts 1 from your score.
  The maximum possible score, given the standard 144 tiles, is 48.
    (From 16 nine-letter words, since 16*9=144.)
  Is it possible to find 16 9-letter words and reach the maximum possible score?
  If not, what's the highest score that can be achieved?

* **Strategy Overview:**
  1. Assess how computationally tractable it is to find long lists if 9-letter words with the available tiles.
    * Start with short candidate lists.  Lengthen as longer solutions are found.
    * While trying to reach a desired wordlist length:
    * Remove some of the words that push the character frequency histogram over its constraints.
    * Add or remove words as needed to get to the desired wordlist length.
    * **Conclusions:**
      * 14-word lists of 9-letter words (that only use available tiles) take a long time to find.
      * 13-word lists can be found in a few minutes.
      * 12-word and 11-word lists can be found almost instantaneously.
  2. Repeatedly find a "stub wordlist" (e.g., an 11-word list), then do an exhaustive search
     to determine whether or not the remaining tiles can all be used to find the other 9-letter words.

* **Strategy Details:**
  * Repeatedly checking psuedo-random 16-word lists to see if they happen to use the available tiles seems doomed.
  * At the other extreme, exhaustively searching through wordlists to see if one uses all available tiles---also doomed.
  * A nice balance is to use 11-word "seed" lists that are "valid" (i.e., stay within the available tiles),
    and then exhaustively search the remaining 5*9=54 tiles to find five more words.  On my laptop, such a search takes
    from 5 minutes to 12 hours, depending on the distribution of letters on the 54 remaining tiles.
  * Searching for the final word in the five-word complementary set is assisted by a WordCache class (see details in script).
    * While one might think that such a cache would improve performance,
      note that almost no time was being spent searching for this final word.

* **UI Notes**
  The program outputs various progress indicators:
     * A single period to denote an evaluation of a candidate list of 11 words.
     * When an 11-word list is found that can be made with available tiles, a timestamp is printed.
     * Following the timestamp are indicators showing progress toward finding a complementary list of five more words.
     * Numbers are printed out to show how many singletons, pairs, triples, quadruples, and quintuples are found.
     * Following the numbers of pairs and triples are letters that represent blocks of computations, with different
       letters used to represent which process/CPU the computation is being done on.

* **Possible future ideas:**
  * [Feature] More flexible command-line options, including selection of challenges, starting conditions, and strategies.
  * [Performance] Be selective about which "stub wordlists" to investigate.
    * E.g., only pursue those whose "gap" frequencies (i.e., the distribution of the remaining tiles)
      closely matches (via least squares or cosine distance) the average letter frequencies of 9-letter
      words in the dictionary.
    * E.g., exclude from consideration those "stub wordlists"
      * those with "gap" frequency histograms that permit too few (e.g., < 100) 9-letter words.
      * those that permit so many 9-letter words that completing would be too computationally expensive.
        [Implemented something like this by adding MAX_PAIR_COUNT]
  * [Performance] Relieve memory pressure caused by large of triples.
    Use backpressure to slow down candidate evaluation in the presence of memory pressure.
  * [Performance, DONE] Parcel out work to different CPUs.
    Did this by "chunking" singletons.  Doing so with pairs/triples/quadruples could have allowed for
    smoother pipelining between phases, but would have had some drawbacks.

**Result! (2016-08-27)**
1.  adjuvants
2.  anonymity
3.  ataraxias
4.  bottoming
5.  bouzoukia
6.  cathected
7.  excelling
8.  jeoparded
9.  overkills
10. prenotify
11. qualifier
12. quavering
13. refurbish
14. towheaded
15. weeweeing
16. zoosperms
 
