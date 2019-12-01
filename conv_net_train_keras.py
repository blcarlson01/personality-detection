import numpy as np
import csv
import joblib
import pickle
import sys
import keras
from keras.layers import Input, concatenate, Dropout, Masking, Bidirectional, TimeDistributed
from keras.layers import Conv3D, MaxPooling3D, Dense, Activation, Reshape, GRU, SimpleRNN, LSTM
from keras.models import Model, Sequential
from keras.activations import softmax
from keras.utils import to_categorical, Sequence
from keras.callbacks import CSVLogger
from keras.callbacks import History, BaseLogger, ModelCheckpoint
import pickle
from pathlib import Path
from keras.callbacks import ModelCheckpoint
import os

import logging

model_name = "main_model"
logging.basicConfig(filename='logger_' + model_name + '.log', level=logging.DEBUG, format='%(asctime)s %(message)s')


class TestCallback(keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % 5 == 0:
            test_data, step_size = self.test_data
            loss, acc = self.model.evaluate_generator(test_data, steps=step_size)
            logging.info('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


class MyLogger(keras.callbacks.Callback):
    def __init__(self, n):
        self.n = n  # logging.info loss & acc every n epochs

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.n == 0:
            curr_loss = logs.get('loss')
            curr_acc = logs.get('acc') * 100
            val_loss = logs.get('val_loss')
            val_acc = logs.get('val_acc')
            logging.info("epoch = %4d  loss = %0.6f  acc = %0.2f%%" % (epoch, curr_loss, curr_acc))
            logging.info("epoch = %4d  val_loss = %0.6f  val_acc = %0.2f%%" % (epoch, val_loss, val_acc))


class Generator(Sequence):

    def __init__(self, x_set, x_set_mairesse, y_set, batch_size, W, sent_max_count, word_max_count, embbeding_size):
        self.x, self.mairesse, self.y = x_set, x_set_mairesse, y_set
        self.batch_size = batch_size
        self.W = W
        self.sent_max_count = sent_max_count
        self.word_max_count = word_max_count
        self.embbeding_size = embbeding_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_m = self.mairesse[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return [make_input_batch(batch_x, W, self.sent_max_count, self.word_max_count, self.embbeding_size), batch_m], \
               to_categorical(batch_y, num_classes=2)


def get_checkpoints(model_dir):
    saved_checkpoints = [f for f in os.listdir(model_dir) if f.startswith('model-' + model_name)]
    saved_checkpoints.sort(reverse=True)
    return saved_checkpoints


def train_conv_net(datasets, W, historyfile, iteration,
		    embbeding_size = 300,
		    n_epochs = 50,
		    batch_size = 50):
    word_max_count = len(datasets[0][0][0])
    sent_max_count = len(datasets[0][0])


    # define model architecture

    model_input = Input(shape=(sent_max_count, word_max_count, embbeding_size, 1), name='main_input')

    # unigrams
    model_1 = Sequential()
    model_1.add(Conv3D(200, (1, 1, embbeding_size), activation='relu',
                       input_shape=(sent_max_count, word_max_count, embbeding_size, 1)))
    model_1.add(MaxPooling3D((1, word_max_count, 1)))

    model_output_1 = model_1(model_input)

    # bigrams
    model_2 = Sequential()
    model_2.add(Conv3D(200, (1, 2, embbeding_size), activation='relu',
                       input_shape=(sent_max_count, word_max_count, embbeding_size, 1)))
    model_2.add(MaxPooling3D((1, word_max_count - 1, 1)))

    model_output_2 = model_2(model_input)

    # trigrams
    model_3 = Sequential()
    model_3.add(Conv3D(200, (1, 3, embbeding_size), activation='relu',
                       input_shape=(sent_max_count, word_max_count, embbeding_size, 1)))
    model_3.add(MaxPooling3D((1, word_max_count - 2, 1)))

    model_output_3 = model_3(model_input)


    model = concatenate([model_output_1, model_output_2, model_output_3], axis=-1)

    after_MaxPooling = MaxPooling3D((sent_max_count, 1, 1))(model)

    mairesse_input = Input(shape=(84,), name='mairesse')
    model = Reshape((600,))(after_MaxPooling)
    concatenated_with_mairsse = concatenate([model, mairesse_input], axis=-1)

    model = Dense(200, activation='sigmoid')(concatenated_with_mairsse)
    model = Dropout(0.5)(model)
    output = Dense(2, activation='softmax')(model)

    final_model = Model(inputs=[model_input, mairesse_input], outputs=output)
    final_model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

    validation_size = int(np.round(0.1 * len(datasets[0])))

    X_train = datasets[0][validation_size:]
    y_train = datasets[1][validation_size:]
    X_validation = datasets[0][:validation_size]
    y_validation = datasets[1][:validation_size]
    X_test = datasets[2]
    y_test = datasets[3]

    mairesse_train = datasets[4][validation_size:]
    mairesse_test = datasets[5]
    mairesse_validation = datasets[4][:validation_size]

    train_data_G = Generator(X_train, mairesse_train, y_train, batch_size, W, sent_max_count, word_max_count,
                             embbeding_size)
    val_data_G = Generator(X_validation, mairesse_validation, y_validation, batch_size, W, sent_max_count,
                           word_max_count,
                           embbeding_size)
    test_data_G = Generator(X_test, mairesse_test, y_test, batch_size, W, sent_max_count, word_max_count,
                            embbeding_size)

    model_dir = 'models/results/' + model_name + '/' + str(iteration)
    checkpoint_path = model_dir + "/model-" + model_name + '-' + str(iteration) + "-{acc:02f}.hdf5"
    # Keep only a single checkpoint, the best over test accuracy.
    checkpoint = ModelCheckpoint(str(checkpoint_path),
                                 monitor='acc',
                                 verbose=1)
    saved_checkpoints = get_checkpoints(model_dir)
    if len(saved_checkpoints) > 0:
        last_checkpoint = saved_checkpoints[0]
        logging.info("Resume training from " + last_checkpoint)
        final_model.load_weights(model_dir + '/' + last_checkpoint)
    else:
        logging.info("Traning from scratch!")
    logging.info(len(X_train) / batch_size)

    history = History()

    final_model.fit_generator(train_data_G, validation_data=val_data_G, steps_per_epoch=len(X_train) / batch_size,
                              validation_steps=len(X_validation) / batch_size, epochs=n_epochs,
                              callbacks=[my_logger, history, checkpoint])
    logging.info("loading best model weights")
    saved_checkpoints = get_checkpoints(model_dir)
    last_checkpoint = saved_checkpoints[0]
    logging.info("Resume weights from " + last_checkpoint)
    final_model.load_weights(model_dir + '/' + last_checkpoint)
    logging.info("evaluating model...")
    loss, acc = final_model.evaluate_generator(test_data_G, steps=len(datasets[0]) / batch_size)
    hist = str(history.history)
    pickle.dump(hist, historyfile)

    logging.info('score = ' + str(loss) + "," + str(acc))
    return loss, acc

def make_input_batch(X_train, W, sent_max_count, word_max_count, embbeding_size):
    size = (len(X_train), sent_max_count, word_max_count, embbeding_size)
    input_train = np.zeros(size)
    for rev_dx, review in enumerate(X_train):
        for sent_idx, sentence in enumerate(review):
            sentence = np.array(sentence)
            indexes = np.where(sentence != 0)[0]
            for idx in indexes:
                input_train[rev_dx][sent_idx][idx] = W[sentence[idx]]
    input_train = input_train.reshape([len(X_train), sent_max_count, word_max_count, embbeding_size, 1])
    return input_train


def make_idx_data_cv(revs, word_idx_map, mairesse, charged_words, cv, per_attr=0, max_l=51, max_s=200, k=300,
                     filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    trainX, testX, trainY, testY, mTrain, mTest = [], [], [], [], [], []
    for idx, rev in enumerate(revs):
        sent = get_idx_from_sent(rev["text"], word_idx_map,
                                 charged_words,
                                 max_l, max_s, k, filter_h)

        if rev["split"] == cv:
            testX.append(sent)
            testY.append(rev['y' + str(per_attr)])
            mTest.append(mairesse[rev["user"]])
        else:
            trainX.append(sent)
            trainY.append(rev['y' + str(per_attr)])
            mTrain.append(mairesse[rev["user"]])
    trainX = np.array(trainX)
    testX = np.array(testX)
    trainY = np.array(trainY)
    testY = np.array(testY)
    mTrain = np.array(mTrain)
    mTest = np.array(mTest)
    return [trainX, trainY, testX, testY, mTrain, mTest]

def get_idx_from_sent(status, word_idx_map, charged_words, max_l=51, max_s=200, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    length = len(status)

    pass_one = True
    while len(x) == 0:
        charged_counter = 0
        not_charged_counter = 0
        for i in range(length):
            words = status[i].split()
            if pass_one:
                words_set = set(words)
                if len(charged_words.intersection(words_set)) == 0:
                    not_charged_counter += 1
                    continue
            else:
                if np.random.randint(0, 2) == 0:
                    continue
            charged_counter += 1
            y = []
            for i in range(pad):
                y.append(0)
            for word in words:
                if word in word_idx_map:
                    y.append(word_idx_map[word])

            while len(y) < max_l + 2 * pad:
                y.append(0)
            x.append(y)
        pass_one = False

    if len(x) < max_s:
        x.extend([[0] * (max_l + 2 * pad)] * (max_s - len(x)))

    return x

if __name__ == "__main__":
    logging.info("loading data...: floatx:")
    my_logger = MyLogger(n=1)
    x = joblib.load("essays_mairesse.p")

    revs, W, W2, word_idx_map, vocab, mairesse = x[0], x[1], x[2], x[3], x[4], x[5]
    logging.info("data loaded!")
    try:
        attr = int(sys.argv[0])
    except IndexError:
        attr = 4

    r = range(0, 10)

    ofile = open('perf_output_' + model_name + "_" + str(attr) + '_w2v.txt', 'w')

    charged_words = []

    emof = open("Emotion_Lexicon.csv", "rt")
    history_file_name = 'history_' + model_name + '_attr_' + str(attr) + '_w2v.txt'
    historyfile = open(history_file_name, 'wb')
    csvf = csv.reader(emof, delimiter=',', quotechar='"')
    first_line = True

    for line in csvf:
        if first_line:
            first_line = False
            continue
        if line[11] == "1":
            charged_words.append(line[0])

    emof.close()

    charged_words = set(charged_words)

    results = []
    for i in r:
        logging.info("iteration = %4d from %4d " % (i, len(r)))
        datasets = make_idx_data_cv(revs, word_idx_map, mairesse, charged_words, i, attr, max_l=149,
                                    max_s=312, k=300,
                                    filter_h=3)

        results = train_conv_net(datasets, W, historyfile, i)
        ofile.write(str(results) + "\n")
        ofile.flush()

    ofile.write(str(results))
    historyfile.close()

