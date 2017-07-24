from dnn_classifier import DNNClassifier
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data")
X_train = mnist.train.images
y_train = mnist.train.labels

X_valid = mnist.validation.images
y_valid = mnist.validation.labels

X_test = mnist.test.images
y_test = mnist.test.labels

dnn = DNNClassifier(tensorboard_logdir="/tensorboard", random_state=42)
dnn.fit(X_train, y_train, n_epochs=100, X_valid=X_valid, y_valid=y_valid)