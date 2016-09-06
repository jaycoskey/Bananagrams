# Bananagram Challenges

Search for a way to group the 144 Bananagram tiles into words of a given length.

**Note:** The scripts in this directory do **not** attempt to play the game Bananagrams.

# 9-letter Bananagram Challenge:

* **Short version:** Find a 16-word list of 9-letter words (16x9), using the 144 Bananagram tiles.
* **Strategy Overview:**
  * Start with an 11-word "seed wordlist" that only uses the available tiles.
  * Repeatedly modify the seed wordlist by removing and adding words, until it stick within the available tiles.
  * Pay particular attention to the rarest tiles.  Specifically, there are only 2 each of: j, k, q, x, z. 
  * For each valid 11-word candidate list, do an exhaustive search for the remaining complementary 5-word list.
  * Use async calls to use multiple processors.
  * Log progress indicators, showing the progress in the search for the complementary 5-word list.
    * Log the number of singletons, pairs, triples, quadruples, and quintuples.
    * In the search for triples and quadruples, show progress for each subprocess executing an async call.
* **Possible future ideas:**
  * Convert from async calls to a continuous data pipeline.
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
 
# 12-letter Bananagram Challenge:

* **Short version:** Find a 12-word list of 12-letter words (12x12), using the 144 Bananagram tiles.
* **Note:** There are far fewer 12-letter words than there are 9-letter words.  It's not even currently
  clear (as of 2016-09-05) if there is a solution.
* **Strategy Overview:**
  * Search more aggressively than in the 9-letter case.
    Populate a cache of 4-word lists.
    I chose a cache size of 100,000,000.
    Search for 8-word lists that have a complementary 4-word list in the cache.
  * Specialize the 8-word lists and 4-word lists as follows.
    * Have the 8-word lists use all the available Bananagram tiles with: j, k, q, x, z, and w.
    * Have the 4-word lists not use any of these letters.
  * The script (bgrams_12x12.py) has three modes of operation:
    * CACHE.  Populate a cache with 4-word lists.
      * Each 4-word list has an integer hash value that can fit in 64-bits.
      * Each 4-word list is stored in a cache files named after the hash value (mod 10K).
    * LOOKUP.  Look up a particular 4-word list from the cache, by its (integer) hash value.
    * SEARCH.  Search for 8-word lists that match a complementary 4-word list already in the cache.
      * First read in the hashes from the cache.  Just the hashes---not the associated 4-word lists.
      * If the letter distribution needed to complement the 8-word list is found in the cache, log the result as a possible solution.
      * Possible solutions can be looked up by hash value in a separate run of the script, using the LOOKUP feature.
      * Due to hash collisions, the resulting 4-word list(s) found might not actually complement the 8-word list.
        * This is why the "solution" is called a "possible solution".

# Bananagram Challenge History
 
* **Challenge:** Here is the original version of the challenge that I heard.
  Use the tiles from the game Bananagrams to form 7-, 8-, and 9-letter words,
  and get the highest score possible, given the following scoring:
  * each 7-letter word is worth 1 point
  * each 8-letter word is worth 2 points
  * each 9-letter word is worth 3 points
  * each unused letter subtracts 1 from your score.
  The maximum possible score, given the standard 144 tiles, is 48.
    (From 16 nine-letter words, since 16*9=144.)
  Is it possible to find 16 9-letter words and reach the maximum possible score?
  If not, what's the highest score that can be achieved?
* I first assessed how computationally tractable the problem was, by finding lists of increasing length.
  * 14-word lists of 9-letter words (that only use available tiles) would take a very long time to find.
  * 13-word lists can be found in a few minutes.
  * 12-word and 11-word lists can be found almost instantaneously.
