# Bananagram Challenges

Search for a way to group the 144 Bananagram tiles into words of a given length.

**Note:** The scripts in this directory do **not** attempt to play the game Bananagrams.

# 9-letter-word Bananagram Challenge:

* **Short version:** Find a 16-word list of 9-letter words (16x9), using the 144 Bananagram tiles.
* **Overview:**
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
* **Result! (2016-08-27)**
  <div class="decimal">
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
  </style>

# 12-letter-word Bananagram Challenge:

* **Short version:** Find a 12-word list of 12-letter words (12x12),
    using the 144 Bananagram tiles.
* **Note:** There are far fewer 12-letter words than there are 9-letter words.
    There might not be a solution.
* **Overview:**
  * I tried some randomized searches without encountering a solution,
    then switched to see how one could (relatively) efficiently search the entire space.
* Stipulative definitions:
    * Rare tiles (2 tiles per letter): q, j, x, z (in order of increasing frequency)
    * Uncommon tiles: w, v, k, f, y, b
    * Common tiles: h, m, p, g, u, c, d, l, o, t, n, r, a, i, s, e
    * Hexlist: A list of 6 integers counting the number of uncommon letters in a given word
    * Decalist: A list of 10 integers counting the number of rare+uncommon letters in a given word
    * Alphalist: A list of 26 integers counting the number of all letters in a given word
* **High-level approach:**
  * Ignore the spelling of the 12-letter words until the last part of the search algorithm.
    Have most of the algorithm deal with letter count distributions, rather than words.
    First, find 10-character distributions, called "decalists".
    I'm still looking for the most efficient way to get to the full 26-character distributions.
    In the final step, convert 26-character count distributions back into actual words.
* **Prologue**
  * Determine by hand what rare letter count groups exist for the "rare tiles".
  * There are nine (9) rare-tile-letter-word-groups.
    * For example,
      a word might have one 'q', but no j, x, or z (e.g., "quintessence"),
      or one 'j' and one 'z', but no q or x (e.g., "jeopardizing").
    * There are eleven (11) different ways to select **eight** words from these nine groups
      so that the eight words have two of each of the rare tiles.
    * Call each one of these 11 ways a "batch".
    * For example, one batch consists of two j-words, two q-words, two xz-words (e.g., "extravaganza" twice),
      and two words without any rare tiles.
    * Now, we're going to shift gears, and think only in terms of letter counts for a while.

| Group |  j  |  q  |  x  |  z  | # of Rare+Common |#words|Example word|
| :---  |---: |---: |---: |---: | ---:             | ---: | :---       |
| A     |  1  |  0  |  0  |  0  | 7                |   28 |conjunctions|
| B     |  1  |  0  |  0  |  1  | 1                |    2 |jeopardizing|
| C     |  0  |  1  |  0  |  0  | 10               |   56 |liquidations|
| D     |  0  |  1  |  0  |  1  | 1                |    6 |tranquilizer|
| E     |  0  |  0  |  1  |  0  | 8                |   94 |ambidextrous|
| F     |  0  |  0  |  1  |  1  | 1                |    1 |extravaganza|
| G     |  0  |  0  |  0  |  2  | 1                |    1 |embezzlement|
| H     |  0  |  0  |  0  |  1  | 9                |  138 |hypnotizable|
| X     |  0  |  0  |  0  |  0  | 57               | 2840 |heaviweights|

**Note:** These numbers are based on the file /usr/share/dict/linux.words in the Cygwin package words-3.0-1.

Eight-word potentially viable combinations from the above groups are called "batches".

|Batch names|Group combos|12-tuples of decalists|
| :---      |:---        | ---:                 |
| alpha     |AA-CC-EE-GX | 22,820,590           |
| bravo     |AA-CC-EE-HH | 31,709,105           |
| charlie   |AA-CC-EF-GX |  6,036,199           |
| delta     |AA-CC-FF-XX | 11,056,266           |
| echo      |AA-CD-EE-GX | 50,474,099           |
| foxtrot   |AA-CD-EF-XX | 37,687,893           |
| golf      |AA-DD-EE-XX | 37,687,893           |
| hotel     |AB-CC-EE-GX | 12,022,392           |
| india     |AB-CC-EF-XX | 72,678,672           |
| juliet    |AB-CD-EE-XX | 98,625,334           |
| kilo      |BB-CC-EE-XX | 75,208,057           |

**Note:** I've attempted to avoid duplicates, but haven't checked
      to see if there are any duplicates in the above decalist count stats.

* **Phase 1**
  * Represent each of the 11 "batches" (see above) by a loop
    with a distinctive set of bounds.  (See the function phase1_batch().)
  * For each Group combination of 8 words that up all the available rare tiles,
    enumerate and sum the possible letter counts for the rare (4 letters) and uncommon (6 letters) letters.
    Subtract this sum from the count of available tiles to find what remains.
    Use cache lookups to find what combinations of uncommon letter counts are possible for the next 4 words.
  * **End result:** One or more files, with each row representing the counts of the rare and uncommon tiles
    of a potential solution to the 12-letter-word Bananagram Challenge.
* **Phase 2**
  * For each batch, take the list of possible counts of the rare and uncommon letters for each of 12 words.
  * **End result:** One or more files, with one each row representing a solution's "signature".
    Each row is list of 12 "alphalists", where an alphalist is a list of 26 integers representing letter counts.
  * **TODO:** Some benchmarking shows that going directly from decalists to alphalists is very computationally expensive.
    Check to see if breaking this jump into smaller steps
    (e.g., from 10-letter counts to 16-letter to 26-letter counts) would be more efficient.
  * **Phase 3**
    * Convert the possible combinations of 26-letter counts (if any) into corresponding word lists.
    * For example, the word "constitution" has these letter counts: c:1, o:2, n:2, s:1, t:3, i:2, ...,
      so the corresponding alphalist would have the integers 1, 2, 2, 1, 3, 2, etc., in some order,
      with plenty of zeros to represent the counts of letters of the alphabet not found in this word.
    * **End result:** Solutions (if any) to the 12x12 Bananagram Challenge.

* **Phase 1 Stats:**

The following times are for a single-threaded run:

| Batch   |Elapsed Time|System Time |
| :---    |  ---:      |   ---:     |
| alpha   | 1h:25m:07s |   1:19:07  |
| bravo   |  1:38:45   |   1:30:49  |
| charlie |  0:21:06   |   0:19:36  |
| delta   |  0:40:30   |   0:37:42  |
| echo    |  0:27:20   |   0:25:17  |
| foxtrot |  2:53:36   |   2:40:49  |
| golf    |  2:07:17   |   1:57:35  |
| hotel   |  0:41:19   |   0:38:16  |
| india   |  4:22:56   |   4:04:12  |
| juliet  |  5:29:27   |   5:03:36  |
| kilo    |  4:15:56   |   3:53:04  |
|**TOTAL**|**24:23:19**|**22:30:03**|

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
* I first assessed how computationally tractable the problem was, by finding wordlists of increasing length.
  * 14-word lists of 9-letter words (that only use available tiles) would take a very long time to find.
  * 13-word lists can be found in a few minutes.
  * 12-word and 11-word lists can be found almost instantaneously.
* Based on this timing information, I settled on an "11+5" strategy:
  Find 11-word lists and, for each one of those, exhaustively search for a 5-word list that complements the 11-word list.

* **Version 0 of bgrams_12x12.py:**
  * Search more aggressively than in the 9-letter case.
    Populate a cache of 4-word lists.
    I chose a cache size of 100,000,000.
    Search for 8-word lists that have a complementary 4-word list in the cache.
  * Specialize the 8-word lists and 4-word lists as follows.
    * Have the 8-word lists use all the available Bananagram tiles with: j, k, q, x, z, and w.
      * Update (2016-09-07): Remove w from this list.
    * Have the 4-word lists not use any of these letters.
      * Update (2016-09-07): Have this list contain the three w's.
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
