#!/usr/bin/python3

import argparse
import sys

PATH_INPUT='/usr/share/dict/words'
PATH_OUTPUT='./bgrams9.txt'
NUM_LETTERS = 9


def write_words(path_input, path_output, num_letters) -> None:
    with open(path_input) as input_file:
        lines = [line.rstrip('\n') for line in input_file]
        words = [line for line in lines
                  if line.isalpha() and line.islower() and len(line) == num_letters]
    with open(path_output, 'w') as output_file:
        for word in words:
            output_file.write(word + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='write_wordlist')
    parser.add_argument('-i', '--input_file',
                        type=argparse.FileType('r'),
                        default=PATH_INPUT,
                        help='specify name of input file, with words of any length')
    parser.add_argument('-o', '--output_file',
                        type=argparse.FileType('a'),
                        default=PATH_OUTPUT,
                        help='specify name of output file, to have words of fixed length')
    parser.add_argument('-n', # '--num_letters',
                        type=int,
                        help='number of letters in words to be selected',
                        # action='store_true'
                        )

    args = parser.parse_args(sys.argv[1:])
    write_words(args.input_file, args.output_file, args.n)
