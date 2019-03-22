from util import *


if __name__ == '__main__':

    # Load Word Vector
    wordvec_file = 'wordvectors/glove.6B.100d.txt'
    wordvec, word2index, index2word = read_wordvec(wordvec_file)

    # Data Process
    train_file = 'data/train.txt'
    test_file = 'data/test.txt'
    train_df, char2index = read_data(train_file)
    test_df, _ = read_data(test_file)

    word_sent, char_sent, tag, word_len, char_len = preprocess(train_df.copy(), word2index, char2index)
    test_word_sent, test_char_sent, test_tag, test_word_len, test_char_len = preprocess(test_df.copy(), word2index, char2index)

    # Save Data
    np.save('data/word_sent.npy', word_sent)
    np.save('data/char_sent.npy', char_sent)
    np.save('data/tag.npy', tag)
    np.save('data/word_len.npy', word_len)
    np.save('data/char_len.npy', char_len)
    np.save('data/wordvec.npy', wordvec)

    with open('data/word2index.json', 'w', encoding='utf-8') as file:
        json.dump(word2index, file)
        
    with open('data/index2word.json', 'w', encoding='utf-8') as file:
        json.dump(index2word, file)
        
    with open('data/char2index.json', 'w', encoding='utf-8') as file:
        json.dump(char2index, file)


    np.save('data/test_word_sent.npy', test_word_sent)
    np.save('data/test_char_sent.npy', test_char_sent)
    np.save('data/test_tag.npy', test_tag)
    np.save('data/test_word_len.npy', test_word_len)
    np.save('data/test_char_len.npy', test_char_len)

