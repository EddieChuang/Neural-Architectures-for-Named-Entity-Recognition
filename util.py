import numpy as np
import pandas as pd
import re, json
import re, os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def read_data(filepath):
    df = pd.DataFrame(columns=['word', 'ne'])  # syntatic chunck, named entity
    char = set()
    nrow = 0
    with open(filepath, 'r', encoding='utf-8') as file:
        
        word, ne = [], []
        for row in file:
            row = row.strip()
            if '-DOCSTART-' in row:
                next(file)
                continue
                
            if row:
                row = row.split(' ')
                word.append(row[0].lower())
                ne.append(row[3])
            else:
                df.loc[nrow] = [word, ne]
                word, ne = [], []
                nrow += 1
    
    char2index = {}
    chars = set(''.join([w for word in df['word'] for w in word] + ['#']))
    for i, char in enumerate(sorted(chars)):
        char2index[char] = i
    char2index.update({'<unk>': len(char2index), '<pad>': len(char2index) + 1})
    return df, char2index


def read_wordvec(wordvec_file):
    
    vocab = set()
    wordvec = []
    with open(wordvec_file, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            line = line.strip().split()
            vocab.add(line[0])
            wordvec.append(line[1:])
    
    vocab.add('<UNK>')
    vocab.add('<PAD>')
    wordvec.extend([[0] * len(wordvec[0])] * 2)
    word2index = {}
    index2word = []
    for i, word in enumerate(sorted(vocab)):
        word2index[word] = i
        index2word.append(word)
    return np.array(wordvec, dtype=np.float32), word2index, index2word
        

def normalize_number(words):
    return [re.sub(r'[0-9]+[\+|\-|,|.|/]?[0-9]+', '0', word) for word in words]


def tag_to_index(tags):
    tag2index = {'B-LOC': 0, 'B-MISC': 1, 'B-ORG': 2, 'B-PER': 3, 'I-LOC': 4, 'I-MISC': 5, 'I-ORG': 6, 'I-PER': 7, 'O': 8}
    return [tag2index[tag] for tag in tags]


def index_to_tag(tags):
    index2tag = ['B-LOC', 'B-MISC', 'B-ORG', 'B-PER', 'I-LOC', 'I-MISC', 'I-ORG', 'I-PER', 'O']    
    return [index2tag[tag] for tag in tags]


def word_to_index(words, word2index):
    unk_word = word2index['<UNK>']
    return [word2index.get(word, unk_word) for word in words]


def char_to_index(words, char2index):
    unk_char = char2index['<unk>']
    return [[char2index.get(char, unk_char) for char in word] for word in words]


def preprocess(train_df, word2index, char2index):
    
    train_df['word'] = train_df['word'].map(normalize_number)
    train_df['char'] = train_df['word'].map(lambda words: [list(word) for word in words])
    train_df['word'] = train_df['word'].map(lambda words: word_to_index(words, word2index))
    train_df['char'] = train_df['char'].map(lambda words: char_to_index(words, char2index))
    train_df['ne']   = train_df['ne'].map(tag_to_index)    
    
    word_len = np.array([len(x) for x in train_df['word']])
    char_en = np.array([[len(word) for word in words] for words in train_df['char'] ])
    
    return train_df['word'].values, train_df['char'].values, train_df['ne'].values, word_len, char_en


def load_train_data():
    word_sent = np.load('data/word_sent.npy')
    char_sent = np.load('data/char_sent.npy')
    tag = np.load('data/tag.npy')
    word_len = np.load('data/word_len.npy')
    char_len = np.load('data/char_len.npy')
    wordvec = np.load('data/wordvec.npy')
    
    with open('data/word2index.json', 'r', encoding='utf-8') as file:
        word2index = json.load(file)
        
    with open('data/index2word.json', 'r', encoding='utf-8') as file:
        index2word = json.load(file)

    with open('data/char2index.json', 'r', encoding='utf-8') as file:
        char2index = json.load(file)
    
    return word_sent, char_sent, tag, word_len, char_len, wordvec, word2index, index2word, char2index


def load_test_data():
    word_sent = np.load('data/test_word_sent.npy')
    char_sent = np.load('data/test_char_sent.npy')
    tag = np.load('data/test_tag.npy')
    word_len = np.load('data/test_word_len.npy')
    char_len = np.load('data/test_char_len.npy')
    wordvec = np.load('data/wordvec.npy')
    
    with open('data/word2index.json', 'r', encoding='utf-8') as file:
        word2index = json.load(file)
        
    with open('data/index2word.json', 'r', encoding='utf-8') as file:
        index2word = json.load(file)

    with open('data/char2index.json', 'r', encoding='utf-8') as file:
        char2index = json.load(file)
    
    return word_sent, char_sent, tag, word_len, char_len, wordvec, word2index, index2word, char2index


def train_val_split(x_train, x_char_train, y_train, seq_len, word_len, train_ratio=.7):
    train_len  = int(len(x_train) * train_ratio)
    train_data = [x_train[: train_len], x_char_train[: train_len], y_train[: train_len], seq_len[: train_len], word_len[: train_len]]
    val_data   = [x_train[train_len: ], x_char_train[train_len: ], y_train[train_len: ], seq_len[train_len: ], word_len[train_len: ]] 
    
    return train_data, val_data


def shuffle_data(data):
    indice = np.arange(len(data[0]))
    np.random.shuffle(indice)
    
    return [d[indice] for d in data]


def next_batch(data, batch_size, word2index, char2index):
    def pad(sequence, max_wlen, pad_token):
        return np.array([seq + [pad_token] * (max_wlen - len(seq)) for seq in sequence])
    
    def char_pad(sequence, max_wlen, max_clen, pad_token):
        pad_seq = []
        for words in sequence:
            pad_words = words + [[pad_token]] * (max_wlen - len(words))            
            pad_seq.append([word + [pad_token] * (max_clen - len(word)) for word in pad_words])
        return np.array(pad_seq)
            

    word_sent, char_sent, tag, word_len, char_len = data[0], data[1], data[2], data[3], data[4] 
    n_batch = len(word_sent) // batch_size
    for i in range(n_batch):
        offset = i * batch_size
        indice = np.arange(offset, offset + batch_size)
        batch_wlen = word_len[indice]
        batch_clen = pad(char_len[indice], max(batch_wlen), 0)
        batch_word = pad(word_sent[indice], max(batch_wlen), word2index['<PAD>'])
        batch_char = char_pad(char_sent[indice], max(batch_wlen), max([max(clen) for clen in batch_clen]), char2index['<pad>'])
        batch_tag  = pad(tag[indice], max(batch_wlen), 8) if tag.any() else []
        
        yield batch_word, batch_char, batch_tag, batch_wlen, batch_clen
    
    
    
    offset = n_batch * batch_size
    if offset == len(word_sent):
        return
    
    batch_wlen = word_len[offset: ]
    batch_clen = pad(char_len[offset: ], max(batch_wlen), 0)
    batch_word = pad(word_sent[offset: ], max(batch_wlen), word2index['<PAD>'])
    batch_char = char_pad(char_sent[offset: ], max(batch_wlen), max([max(clen) for clen in batch_clen]), char2index['<pad>'])
    batch_tag  = pad(tag[offset: ], max(batch_wlen), 8) if tag.any() else []

    yield batch_word, batch_char, batch_tag, batch_wlen, batch_clen


def get_entities(sequence_tag):
    
#     entity = {
#         'begin': 0,
#         'end': 0,
#         'type': '' 
#     }
    entities = []
    is_ne = False
    ne_type = ''
    for i, tag in enumerate(sequence_tag):
        if is_ne and ('B-' in tag or 'O' in tag):
            entities.append({'begin': begin, 'end': i, 'type': ne_type}) 
            is_ne = False
        if 'B-' in tag:
            begin = i
            is_ne = True
            ne_type = tag.split('-')[1]
        elif 'I-' in tag and ne_type != tag.split('-')[1]:
            is_ne = False
    if is_ne:
        entities.append({'begin': begin, 'end': i, 'type': ne_type})
        
    return entities     
        

if __name__ == '__main__':
    pass