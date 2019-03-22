from tensorflow.nn import  bidirectional_dynamic_rnn
from tensorflow.nn.rnn_cell import LSTMCell, DropoutWrapper

def BiLSTM(sequence, seq_len, unit, name):
    cell_fw = LSTMCell(unit, name='fw' + name)
    cell_bw = LSTMCell(unit, name='bw' + name)
#     cell_fw = DropoutWrapper(cell_fw,  output_keep_prob=0.5)
#     cell_bw = DropoutWrapper(cell_bw,  output_keep_prob=0.5)
    
    (outputs, states) = bidirectional_dynamic_rnn(cell_fw, cell_bw, sequence, seq_len, dtype=tf.float32)
    # outputs: (output_fw, output_bw), both with shape (batch_size, max_len, unit)
    # states:  ((cell_fw, state_fw), (cell_bw, state_bw)), fw & bw final state with shape (batch_size, unit)
    return outputs, states


if __name__ == '__main__':
    pass