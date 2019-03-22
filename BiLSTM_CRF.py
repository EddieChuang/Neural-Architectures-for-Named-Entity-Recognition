import tensorflow as tf
from tensorflow.nn import embedding_lookup, bidirectional_dynamic_rnn, dropout
from tensorflow.nn.rnn_cell import LSTMCell, DropoutWrapper
from tensorflow.contrib.crf import crf_log_likelihood, viterbi_decode
from tqdm import tqdm
from pprint import pprint
import numpy as np
import os
from util import *
from layers import *


class NERTagger:
    def __init__(self, wordvec, config):
        '''
        ##### placeholder #####
        word_sent: word-level sentence, (batch_size, max_word_len)
        char_sent: char-level sentence, (batch_size, max_word_len, max_char_len)
        word_len:  sentence length, (batch_size, )
        char_len:  word length, (batch_size, max_word_len)
        tag: answer tag, (batch_size, max_word_len)
        '''
        tf.reset_default_graph()
        self.config = config
        
        self.hidden_unit = config['hidden_unit']
        self.context_lstm_unit = config['context_lstm_unit']
        self.char_vocab_size = config['char_vocab_size']
        self.char_emb_dim = config['char_emb_dim']
        self.char_lstm_unit = config['char_lstm_unit']
        self.num_class = config['num_class']
        self.learning_rate = config['learning_rate']
        
        self.word_sent = tf.placeholder(tf.int32, (None, None)) 
        self.char_sent = tf.placeholder(tf.int32, (None, None, None))
        self.word_len  = tf.placeholder(tf.int32, (None, ))
        self.char_len  = tf.placeholder(tf.int32, (None, None))
        self.tag = tf.placeholder(tf.int32, (None, None))
        
        
        self.w = tf.get_variable('hidden_weight', (self.context_lstm_unit * 2, self.hidden_unit)) 
        self.b = tf.get_variable('hidden_bias', (self.hidden_unit, ))
        
        self.word_embedding = tf.get_variable(name='word_embedding', 
                                              shape=wordvec.shape, 
                                              initializer=tf.constant_initializer(wordvec, dtype=tf.float32),
                                              dtype=tf.float32,
                                              trainable=config['word_emb_trainable'])
        
        
        self.char_embedding = tf.get_variable(name='char_embedding', 
                                              shape=(self.char_vocab_size, self.char_emb_dim), 
                                              initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0),
                                              dtype=tf.float32,
                                              trainable=True)
        
    def embedding_layer(self, word_sent, char_sent, char_len):
        '''
        word_sent: word-level sentence, (batch_size, max_word_len)
        char_sent: char-level sentence, (batch_size, max_word_len, max_char_len)
        word_len:  sentence length, (batch_size, )
        char_len:  word length, (batch_size, max_word_len)
        word_embedding: word embedding matrix, (word_vocab_size, word_emb_dim)
        '''
        max_word_len = tf.shape(word_sent)[1]
        max_char_len = tf.shape(char_sent)[2]
        
        word_emb = embedding_lookup(self.word_embedding, word_sent)  # (batch_size, max_word_len, word_emb_dim)
        char_emb = embedding_lookup(self.char_embedding, char_sent)  # (batch_size, max_word_len, max_char_len, char_emb_dim)

        
        # Reshape char_emb to 3D tensor for BiLSTM input
        char_emb = tf.reshape(char_emb, (-1, max_char_len, self.char_emb_dim))  # (batch_size * max_word_len, max_char_len, char_emb_dim)
        char_len = tf.reshape(char_len, (-1, ))   # (batch_size * max_word_len, )
        
        # Get final states which represent char-level representation
        _, states = BiLSTM(char_emb, char_len, self.char_lstm_unit, 'char_embedding')
        final_h = [states[0][1], states[1][1]]  # [forward final state, backward final state]
        char_emb = tf.concat(final_h, axis=1)  # (batch_size * max_word_len, char_lstm_unit * 2)
        
        # Reshape char_emb to match word_emb's shape for concatenating both tensors
        char_emb = tf.reshape(char_emb, (-1, max_word_len, self.char_lstm_unit * 2))  # (batch_size, max_word_len, char_lstm_unit * 2)
        
        # Token Representation
        emb = tf.concat([word_emb, char_emb], axis=2)  # (batch_size, max_word_len, word_emb_dim + char_lstm_unit * 2)
        emb = dropout(emb, keep_prob=0.5)
        return emb
    
    
    def build(self):
        ##### Context Representation #####
        # emb_dim = word_emb_dim + char_lstm_unit * 2
        word_rep   = self.embedding_layer(self.word_sent, self.char_sent, self.char_len)   # (batch_size, max_word_len, emb_dim)
        outputs, _ = BiLSTM(word_rep, self.word_len, self.context_lstm_unit, 'context_representation') 
        context    = tf.concat(outputs, axis=2)  # (batch_size, max_word_len, context_lstm_unit * 2)
        
        ##### Hidden Layer #####
        max_word_len = tf.shape(context)[1]
        context      = tf.reshape(context, (-1, self.context_lstm_unit * 2))   # (batch_size * max_word_len, context_lstm_unit * 2)
        dense        = tf.matmul(context, self.w) + self.b                     # (batch_size * max_word_len, hidden_unit)
        self.scores  = tf.reshape(dense, (-1, max_word_len, 100))   # (batch_size, max_word_len, hidden_unit)
        
        ##### CRF #####
        log_likelihood, self.transition_params = crf_log_likelihood(self.scores, self.tag, self.word_len)
        self.loss = tf.reduce_mean(-log_likelihood)
        
#         self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

                
    
    def fit(self, train_data, val_data, epoch_size, batch_size, word2index, char2index, model_name):
        def learn(data, epoch, mode):
            tn = tqdm(total=len(data[0]))
            nbatch, epoch_loss, epoch_acc = 0, 0, 0 
            for batch_word, batch_char, batch_tag, batch_wlen, batch_clen in next_batch(data, batch_size, word2index, char2index):
                feed_dict = {
                    self.word_sent: batch_word,
                    self.char_sent: batch_char, 
                    self.word_len: batch_wlen,
                    self.char_len: batch_clen,
                    self.tag: batch_tag
                }
                if mode == 'train':
                    fetches = [self.loss, self.optimizer]
                    loss, _ = self.sess.run(fetches, feed_dict)
                    tn.set_description('Epoch: {}/{}'.format(epoch, epoch_size))
                elif mode == 'validate':                    
                    fetches = [self.loss]
                    loss = self.sess.run(fetches, feed_dict)[0]
                
                tn.set_postfix(loss=loss, mode=mode)
                tn.update(n=len(batch_word))
                
                epoch_loss += loss
                nbatch += 1
            
            tn.set_postfix(loss=epoch_loss/nbatch, mode=mode)
            return [epoch_loss/nbatch]
                
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
                
        train_log, val_log = [], []
        print('Train on {} samples, validate on {} samples'.format(len(train_data[0]), len(val_data[0])))
        for epoch in range(1, epoch_size + 1):       
            train_data = shuffle_data(train_data)
            # train
            train_log.append(learn(train_data, epoch, 'train'))

            # validate
            if len(val_data[0]) > 0:
                val_log.append(learn(val_data, epoch, 'validate')) 
        
        self.save(model_name, train_log, val_log)
    
    
    def predict(self, data, word_to_index, char2index):
        
        tn = tqdm(total=len(data[0]))
        batch_size = 100
        transition_params = self.transition_params.eval(session=self.sess)
        prediction = []
        for batch_word, batch_char, batch_tag, batch_wlen, batch_clen in next_batch(data, batch_size, word2index, char2index):
            fetches = [self.scores]
            feed_dict = {
                self.word_sent: batch_word,
                self.char_sent: batch_char, 
                self.word_len: batch_wlen,
                self.char_len: batch_clen
            }
            scores = self.sess.run(fetches, feed_dict)[0]
            scores = [score[:wlen] for score, wlen in zip(scores, batch_wlen)]
            prediction.extend([viterbi_decode(score, transition_params)[0] for score in scores])
            
            tn.set_postfix(mode='predict')
            tn.update(n=len(batch_word))
        
        return np.array(prediction)
    
    
    def save(self, model_name, train_log, val_log):
        model_dir = 'models/{}'.format(model_name)
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
            os.mkdir('{}/result'.format(model_dir))
        
        # save model
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, '{}/{}.ckpt'.format(model_dir, model_name))
        
        # save config
        with open('{}/config.json'.format(model_dir), 'w', encoding='utf-8') as file:
            json.dump(self.config, file)
            
        # save log
        with open('{}/log'.format(model_dir), 'w', encoding='utf-8') as file:
            for i in range(len(train_log)):
                tlog = train_log[i]
                vlog = val_log[i] if len(val_log) > 0 else []
                log_str = 'Epoch {}: train_loss={}'.format(i+1, tlog[0])
                log_str += ', val_loss={}'.format(vlog[0]) if vlog else ''
                file.write(log_str + '\n')
            
        print('Model was saved in {}'.format(save_path))
    
    
    def restore(self, model_path):
        saver = tf.train.Saver()
        self.sess = tf.Session()
        saver.restore(self.sess, model_path)


if __name__ == '__main__':
    pass