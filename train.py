from util import *
from BiLSTM_CRF import NERTagger
from metrics import accuracy

if __name__ == '__main__':

    # Load train data
    word_sent, char_sent, tag, word_len, char_len, wordvec, word2index, index2word, char2index = load_train_data()
    train_data, val_data = train_val_split(word_sent, char_sent, tag, word_len, char_len, train_ratio=.8)

    # Load wordvec
    wordvec_file = 'wordvectors/glove.6B.100d.txt'
    wordvec, word2index, index2word = read_wordvec(wordvec_file)

    # hyperparameter setting
    epoch_size = 10
    batch_size = 1
    config = {
        'num_class': 9,
        'char_lstm_unit': 25,
        'context_lstm_unit': 100,
        'hidden_unit': 100,
        'word_emb_dim': len(wordvec[0]),
        'char_emb_dim': 25,
        'word_vocab_size': len(wordvec),
        'char_vocab_size': len(char2index),
        'learning_rate': 1e-2,
        'wordvec': wordvec_file,
        'word_emb_trainable': True,
        'epoch_size': epoch_size,
        'batch_size': batch_size
    }

    # train ner tagger
    tagger = NERTagger(wordvec, config)
    tagger.build()
    model_name = 'model'
    tagger.fit(train_data, val_data, epoch_size, batch_size, word2index, char2index, model_name)       


    # predict test data
    test_word_sent, test_char_sent, test_tag, test_word_len, test_char_len, _, _, _, _ = load_test_data()
    test_data = [test_word_sent, test_char_sent, test_tag, test_word_len, test_char_len]      


    prediction = tagger.predict(test_data, word_to_index, char2index)
    print('accuracy: ' + accuracy(pd.Series(prediction).map(index_to_tag), pd.Series(test_data[2]).map(index_to_tag)))

