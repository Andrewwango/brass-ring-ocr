"""
Functions for word matching using fuzzy matching
"""

from fuzzywuzzy import process
from itertools import permutations

def hardcoded_rulebook(candidates, confidences):
    """Rules to pick best candidate word transcription given confidences and hardcoded rules
    Add custom rules into if statements below.

    Args:
        candidates (list): list of candidate transcriptions
        confidences (_type_): each candidate's confidence \in [0,1]

    Returns:
        str: best candidate
    """
    if "win" in candidates or "nim" in candidates:
        return "win"
    elif confidences[0] > confidences[1]:
        return candidates[0]
    else:
        return candidates[1]

def process_words(lines, batch_size):
    """Process words returned by DTRB log

    Args:
        lines (list of lists): batch of transcriptions returned from utils.read_log
        batch_size (int): number of images in batch

    Returns:
        list: list of words for each image, where words is a list of transcribed words from each image
    """
    words_batch = [[] for _ in range(batch_size)]

    for i in range(int(len(lines)/2)):
        img_n = int(lines[i*2][0].split("_")[2])

        [[w1, c1], [w2, c2]] = [[lines[i*2][1], float(lines[i*2][2])], [lines[i*2+1][1], float(lines[i*2+1][2])]]

        words_batch[img_n] += [hardcoded_rulebook([w1, w2], [c1, c2])]

    return words_batch

def match_words(words, classes):
    """Match every permutation of list of words to list of matches with fuzzy matching
    Probably doesn't need to be permutations, best to do just every starting position but keep the order,
    but fuzzy matching is so fast it doesn't matter.

    Args:
        words (list): list of words identified in the image
        classes (list): list of classes to compare to the words (e.g. winusa556)

    Returns:
        str: best possible match using fuzzy matching
        int: matching score, where 100 indicates full match
    """
    pred = ""; conf = 0
    for query in ["".join(t) for t in permutations(words)]:
        p, c = process.extractOne(query, classes)
        if c > conf:
            conf = c
            pred = p

    return pred, conf

def match_batch(words_batch, classes, conf_thresh=60, reject_word="REJECT"):
    """Match batch of words and use OOD detection

    Args:
        words_batch (list): list of words lists, for matching
        classes (list): list of classes to compare to the words (e.g. winusa556)
        conf_thresh (int, optional): Percentage threshold for matching scores, under which they are rejected. Defaults to 60.
        reject_word (str, optional): text to return when rejected (i.e. either OOD or failure). Defaults to "REJECT".

    Returns:
        list: best matches
        list: match scores
    """
    preds = []
    confs = []
    for words in words_batch:
        pred, conf = match_words(words, classes)
        confs += [conf]
        if conf <= conf_thresh:
            preds += [reject_word]
        else:
            preds += [pred]
    return preds, confs