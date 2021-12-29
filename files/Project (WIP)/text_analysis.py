# Put text from database into the mood analyser and other text analysis. This is a WIP.
#Plots/Dashboard (To do): timeline for mood, statistics, attention weights (highlight important words) 

from save_data import db_manager
import datetime
import pandas as pd
import numpy as np
import re

def make_bitesize(text, bitesize = 30, mode='naive', step = None, pad = False, padding=' ', sep='[.?!]'):
    '''
    Divides a string of text into an array of smaller snippets of the same text.\n
    text: the long string of text to divide up\n
    bitesize: desired length of new string snippets\n
    mode: 'naive' for chopping up the string with a given length\n
          'full' for chopping the string into full sentences. Mode 'full' ignores strings of length 1 and less.\n
    step: The stepsize when traversing the string. Must be integer (>= 0) or None. Using 0 or None result in no overlap.\n
    pad: Whether or not to apply padding if a snippet is shorter than the bitesize.\n
    padding: What char to use for padding.\n
    sep: Regular expression used by the 'full' mode to find sentence ends.
    '''
    if step==None:
        step=0

    assert bitesize > 0
    assert step >= 0

    a = []

    if len(text) <= bitesize:
        if pad:
            s = pad_string(text, bitesize, padding)
            return a.append(s)
        else:
            return a.append(text)

    if mode=='naive':
        s = ''
        N = len(text)

        if step == 0:
            for n in range(0, N, bitesize):
                s = text[n: n+bitesize]
                if pad and len(s) < bitesize:
                    s = pad_string(s, bitesize, padding)
                a.append(s)
        else:
            for n in range(0, N, step):
                s = text[n: n+bitesize]
                if pad and len(s) < bitesize:
                    s = pad_string(s, bitesize, padding)
                a.append(s)
                    
    elif mode=='full':
        punctuation = re.finditer(sep, text)
        start = 0
        for p in punctuation:
            end = p.end()
            s = text[start:end]

            if len(s) > 1:
                if pad and len(s) < bitesize:
                    s = pad_string(s, bitesize, padding)
                    
                a.append(s)
            start = end
    else:
        raise ValueError('Choose mode \'naive\' or \'full\'.')
    return a

def pad_string(text, size, padding=' '):
    '''Pads a string to desired siz.e\n
        text: The string to be padded\n
        size: The new size.\n
        padding: The char to pad with.
    '''
    return '{snippet:{c}<{width}}'.format(snippet=text, c = padding, width=size)

def hist_words(text, remove_apostrophe=True):
    '''Count the words in the text to e.g. make a histogram from.'''
    
    scrubbed_text = ''
    #Remove (), punctuation, comma, ?, !, and preserve word-word
    if remove_apostrophe:
        scrubbed_text = ''.join(re.findall(r'(?:[\w+ /]|[/-]\w)+', text.lower()))
    else:
        scrubbed_text = ''.join(re.findall(r'(?:[\w+ /]|[/\'-]\w)+', text.lower()))

    # Unjoin words bound to /
    for c in re.finditer('[/]', scrubbed_text):
        pre_string = scrubbed_text[0: c.start()]
        tail = scrubbed_text[c.end():]

        scrubbed_text = pre_string + ' ' + tail

    words = scrubbed_text.split()
    unique_words = np.unique(words)
    word_count = []
    for uw in unique_words:
        word_count.append( (uw, sum( [ 1 for word in words if word == uw ] ) ) )

    return word_count, len(words)

def test_functions():
    db = db_manager()

    tab_names = db.list_table_names()

    raw_data = db.peek(tab_names[1], coloumns='datetime [DATE], time, main_class, likes, opinion')

    data = pd.DataFrame(raw_data, columns=['Date', 'Time', 'Likes', 'Main', 'Opinion'])

    crumbs = make_bitesize(data['Opinion'][0], mode='naive', step=10, pad = True)
    hist = hist_words(data['Opinion'][0])
    print(' ')


if __name__ ==  '__main__':
    test_functions()