import tensorflow as tf


BATCH_SIZE = 200
SEQUENCE_SIZE = 100
LEARNING_RATE = 1e-3

with open('casmurro.txt') as f:
    text = f.read()

vocab_to_index = {t: i for i, t in enumerate(set(text))}
index_to_vocab = {v: k for k, v in vocab_to_index.items()}
vocab_size = len(vocab_to_index)

text_index = [vocab_to_index[t] for t in text]


x = tf.placeholder(tf.uint8, [BATCH_SIZE, SEQUENCE_SIZE], name='x')
y = tf.placeholder(tf.uint8, [BATCH_SIZE, SEQUENCE_SIZE], name='y')

with tf.name_scope('lstm'):
    rnn_layers = [tf.nn.rnn_cell.BasicLSTMCell(size) for size in [128, 256]]
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    initial_state = multi_rnn_cell.zero_state(BATCH_SIZE, tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(
        cell=multi_rnn_cell,
        inputs=tf.one_hot(x, vocab_size),
        initial_state=initial_state)

with tf.name_scope('output'):
    w = tf.Variable(tf.truncated_normal((256, vocab_size), stddev=0.1), name='weights')
    b = tf.Variable(tf.zeros(vocab_size), name='bias')
    
    logits = tf.add(tf.matmul(x, w), b, name='logits')
    softmax = tf.nn.softmax(logits, name='predictions')

with tf.name_scope('loss'):
    y_one_hot = tf.one_hot(y, vocab_size)
    loss = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=y_one_hot)
    
    step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    feed_dict = {x: text, y: asdasd}
    sess.run([loss, step])