import joblib
from src.preprocessing import norm_names
from src.definitions import SMALL_DATA_PATH, SEQUENCE_LENGTH, WORD2VEC_DIM, FULL_DATA_PATH, FULL_W2V_X_TEST_PATH, FULL_W2V_X_TRAIN_PATH,\
    SMALL_W2V_X_TRAIN_PATH, SMALL_W2V_X_TEST_PATH, MEDIUM_DATA_PATH, MEDIUM_W2V_X_TRAIN_PATH, MEDIUM_W2V_X_TEST_PATH,ALL_CLASSES,\
    FULL_TFIDF_TEST_PATH, FULL_TFIDF_TRAIN_PATH
from keras.utils import to_categorical
import tensorflow as tf
import numpy
from sklearn.metrics import classification_report
tf.logging.set_verbosity(tf.logging.INFO)



def main():
    all_data = joblib.load(FULL_DATA_PATH)
    y_train = to_categorical(norm_names(all_data['y_train']),19)
    y_test = norm_names(all_data['y_test'])
    X_train = joblib.load(FULL_TFIDF_TRAIN_PATH)
    X_test = joblib.load(FULL_TFIDF_TEST_PATH)
    mdl = LogisticRegression()
    mdl.train(X_train, y_train, X_test, y_test)


class LogisticRegression(object):
    BATCH_SIZE = 1

    def __init__(self):
        self.model_parameters = None
        self.outputs = None
        self.activations = None
        self.model_loss = None
        self.optimizer = None
        self.epoch_count  = None
        self.train_data = None
        self.train_targets = None
        self.init = None
        self.test_targets = None
        self.test_data = None

    def tf_init(self, input_dimensionality, targets_count):
        self.model_parameters = dict()
        self.model_parameters['average_vector'] = tf.placeholder(tf.float64, [None, input_dimensionality])
        self.model_parameters['target_class_matrix'] = tf.placeholder(tf.float64, [None, targets_count])
        self.model_parameters['weights_matrix'] = tf.Variable(tf.zeros([input_dimensionality, targets_count],
                                                                       dtype=tf.float64))
        self.model_parameters['bias_matrix'] = tf.Variable(tf.zeros([targets_count], dtype=tf.float64))
        self.model_parameters['global_step_tensor'] = tf.Variable(0, name='global_step', dtype=tf.int64)
        return self.model_parameters

    def construct_model(self):
        bias = self.model_parameters['bias_matrix']
        weights = self.model_parameters['weights_matrix']
        inputs = self.model_parameters['average_vector']
        targets = self.model_parameters['target_class_matrix']
        self.outputs = tf.matmul(inputs, weights) + bias
        self.activations = tf.nn.softmax(self.outputs)
        self.model_loss = tf.reduce_mean(-tf.reduce_sum(targets*tf.log(self.activations), axis=1))
        self.optimizer = tf.train.AdagradDAOptimizer(learning_rate=1, global_step=self.model_parameters['global_step_tensor']).minimize(self.model_loss)

    def train_session_start(self):
        with tf.Session() as current_session:
            saver = tf.train.Saver()
            current_session.run(self.init)
            previous_accuracy = 0
            for epoch in range(self.epoch_count):
                average_cost = 0.
                total_batch = int(self.train_data.shape[0]/self.BATCH_SIZE)
                for i in range(total_batch):
                    start_index = i*self.BATCH_SIZE
                    end_index = (i+1)*self.BATCH_SIZE
                    data_batch = self.train_data[start_index:end_index].toarray()
                    targets_batch = self.train_targets[start_index:end_index]
                    _, cost = current_session.run([self.optimizer, self.model_loss],
                                               feed_dict={self.model_parameters['average_vector']: data_batch,
                                                          self.model_parameters['target_class_matrix']: targets_batch})
                    average_cost += cost / total_batch
                    print(i, cost)
                if (epoch + 1) % 1 == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(average_cost))
                    correct_prediction = tf.equal(tf.argmax(self.activations, 1), tf.argmax(self.model_parameters['target_class_matrix'], 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
                    average_acc = 0.
                    total_batch = int(self.test_data.shape[0] / self.BATCH_SIZE)
                    pred = []
                    for i in range(total_batch):
                        start_index = i * self.BATCH_SIZE
                        end_index = (i + 1) * self.BATCH_SIZE
                        data_batch = self.test_data[start_index:end_index].toarray()
                        targets_batch = self.test_targets[start_index:end_index]
                        new_accuracy = accuracy.eval({self.model_parameters['average_vector']: data_batch,
                                                      self.model_parameters[
                                                          'target_class_matrix']: targets_batch})
                        pred += list(tf.argmax(self.activations, 1).eval({self.model_parameters['average_vector']: data_batch}))
                        average_acc += new_accuracy / total_batch
                        print(i, new_accuracy)
                    print("Accuracy:", average_acc)
                    print(classification_report(self.true_test_targets, pred,
                                                target_names=ALL_CLASSES))
                    if average_acc > previous_accuracy:
                        previous_accuracy = average_acc
                    else:
                        print(classification_report(self.true_test_targets, pred,
                                                    target_names=ALL_CLASSES))
                        saver.save(current_session, 'D:\\usr\\gwm\\pyprojects\\automatic_detection_of_russian_text_genre\\trained_models\\log_reg.hdf5')

                        break

    def train(self, train_data, train_targets, test_data, test_targets, epoch_count=100):
        self.epoch_count = epoch_count
        self.train_data = train_data
        self.train_targets = train_targets
        self.test_data = test_data
        self.test_targets = to_categorical(test_targets, len(ALL_CLASSES))
        self.true_test_targets = test_targets
        self.tf_init(train_data.shape[1], len(ALL_CLASSES))
        self.construct_model()
        self.init = tf.global_variables_initializer()
        self.train_session_start()


if __name__ == '__main__':
    main()
