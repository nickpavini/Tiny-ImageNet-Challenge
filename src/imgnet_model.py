"""
    Class used to easily declare Neural Network models and train, validate and
    test the network. Class also records metrics/graphs based on loss function
    as well as a few other metrics/graphs.
"""
import tensorflow as tf
from imgnet_dataset import Dataset # Dataset handle
import sys, os # command arguments, path management
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #disables AVX/FMA warning
import imgnet_constants as consts # const info regarding dataset
import random # for id generation
from collections import OrderedDict #dictionary for holding network
from tqdm import tqdm

class Model: # /path/to/hdf5_file, pos. int for getting train/val/test batches, amount of train photos/labels/filenames to put in ram
    def __init__(self, hdf5_file, batch_size, chunk_size,
        conv_layers = None, #must provide a shape that each dim will specify features per layer.. ex. [32,64,64] -> 3 layers, filters of 32, 64, and 64 features
        conv_kernels = None, #must provide a shape that will specify kernel dim per layer.. ex. [3,5,5] -> 3x3x3 5x5x5 and 5x5x5 filters.. must have same num of dimenions as conv_layers
        pool_layers = None, #must provide a shape that each dim will specify filter size.. ex. [2,2,2] -> 3 pool layers, 2x2x2 filters and stride of 2 is always
        dropout_layers = None, #must be a shape where each dimension is the probability a neuron stays on or gets turned off... must check... ex. [.4,.4,.4] -> 3 layers with keep probability of 0.4
        fc_layers = None, #must provide a shape that each dim will specify units per connected layer.. ex. [1024,256,1] -> 3 layers, 1024 units, 256, units and 1 unit... last fully connected is the logits layer
        loss_function = None, #must be a tensorflow loss function
        optimizer = None, #must be a tensorflow optimizing function with a learning rate already... see unit tests example below
        ordering = None, #must be a string representing ordering of layers by the standard of this class... ex. "cpcpff" -> conv, max_pool, conv1, max_pool, fully connected, fully connected.. and the num of characters must match the sum of all of the dimensions provided in the layers variables
        storage_dir = None, #complete path to an existing directory you would like model data stored
        gpu_mode = False, #booling for whether or not to enable gpu mode
        id = None #provide a model id for testing/training an existing model
    ):
        if (id):
            assert (os.path.isdir(os.path.join(os.path.join(storage_dir, str(id)), 'tmp'))), ("Unable to locate model " + str(id) + " in the specified storage folder" + os.path.isdir(os.path.join(os.path.join(storage_dir, str(id)), 'tmp')))
            self.id = id
            self.existing_model = True
        else:
            self.id = random.randint(100000, 999999) #generate random 6-digit model id
            self.existing_model = False

        assert (len(conv_layers) + len(pool_layers) + len(dropout_layers) + len(fc_layers) == len(ordering)), "Number of layers does not equal number of entries in the ordering list."
        self.id_dir = os.path.join(storage_dir, str(self.id)) # /dir/to/storage/id
        self.model_dir = os.path.join(self.id_dir, 'tmp') # dir to store model
        self.logs_dir = os.path.join(self.id_dir, 'logs') # dir to store metrics and graphs and such
        None if os.path.isdir(self.id_dir) else os.makedirs(self.id_dir) #create dir if need be
        None if os.path.isdir(self.model_dir) else os.makedirs(self.model_dir) #create id dir if need be
        None if os.path.isdir(self.logs_dir) else os.makedirs(self.logs_dir) #create id dir if need be
        self.dataset = Dataset(hdf5_file, batch_size, chunk_size)
        self.conv_layers = conv_layers
        self.conv_kernels = conv_kernels
        self.pool_layers = pool_layers
        self.dropout_layers = dropout_layers
        self.fc_layers = fc_layers
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.ordering = ordering.lower() #convert all to lowercase for simplicity
        self.gpu_mode = gpu_mode
        self.flattened = False #flag to know if we have already flattened the data once we come to fully connected layers
        self.network_built = False #flag to see if we have already built the network
        self.epochs = 0 #number of epochs we have currently completed successfully with increasing validation accuracy
        self.stop_threshold = 10 #number of epochs that the network should check for an improvement in validation accuracy before


    #2d conv with relu activation
    def conv_2d(self, inputs, filters, kernel_size, name=None):
        out = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size,
                                 padding='same', activation=tf.nn.relu,
                                 name=name)
        return out

    #average pooling with same padding
    def avg_pool2d(self, inputs, pool_size, name=None):
        out = tf.layers.average_pooling2d(inputs, pool_size=pool_size, strides=pool_size,
                                        padding='same', name=name)
        return out

    #max pooling with same padding
    def max_pool2d(self, inputs, pool_size, name=None):
        out = tf.layers.max_pooling2d(inputs, pool_size=pool_size, strides=pool_size,
                                        padding='same', name=name)
        return out

    #n-dimensions to 1-dimension
    def flatten(self, inputs):
        out = tf.contrib.layers.flatten(inputs)
        return out

    #fully connected layer with relu activation
    def dense_relu(self, inputs, units, name=None):
        out = tf.layers.dense(inputs, units, activation=tf.nn.relu,
                                name=name)
        return out

    #fully connected no relu, or logits layer
    def dense(self, inputs, units, name=None):
        out = tf.layers.dense(inputs, units,
                                name=name)
        return out

    #dynamically build the network
    def build_network(self):
        self.network = OrderedDict({'labels': tf.placeholder(tf.float32, [None, consts.CLASSIFICATIONS])}) #start a dictionary with first element as placeholder for the labels
        self.network.update({'inputs': tf.placeholder(tf.float32, [None, consts.SHAPE[0], consts.SHAPE[1], consts.SHAPE[2]])}) #append placeholder for the inputs
        c_layer, p_layer, d_layer, h_layer, a_layer = 0, 0, 0, 0, 0 #counters for which of each type of layer we are on

        #append layers as desired

        for command in self.ordering: #for each layer in network
            if command == 'c': #convolution
                shape = (self.conv_kernels[c_layer], self.conv_kernels[c_layer]) #convert dim provided into a tuple
                self.network.update({'conv'+str(c_layer): self.conv_2d(self.network[next(reversed(self.network))], self.conv_layers[c_layer], shape, 'conv'+str(c_layer))}) #append the desired conv layer
                c_layer += 1
            elif command == 'p': #max_pooling
                shape = (self.pool_layers[p_layer], self.pool_layers[p_layer])
                self.network.update({'max_pool'+str(p_layer): self.max_pool2d(self.network[next(reversed(self.network))], shape, 'max_pool'+str(p_layer))})
                p_layer += 1
            elif command == 'd': #dropout
                self.network.update({'dropout'+str(d_layer): tf.nn.dropout(self.network[next(reversed(self.network))], self.dropout_layers[d_layer])})
                d_layer += 1
            elif command == 'a': #average pooling
                shape = (self.pool_layers[a_layer], self.pool_layers[a_layer])
                self.network.update({'avg_pool'+str(a_layer): self.max_pool2d(self.network[next(reversed(self.network))], shape, 'avg_pool'+str(a_layer))})
            elif command == 'h': #fully connected
                if self.flattened:
                    self.network.update({'fc'+str(h_layer): self.dense_relu(self.network[next(reversed(self.network))], self.fc_layers[h_layer], 'fc'+str(h_layer))})
                else:
                    self.network.update({'fc'+str(h_layer): self.dense_relu(self.flatten(self.network[next(reversed(self.network))]), self.fc_layers[h_layer], 'fc'+str(h_layer))})
                    self.flattened = True
                h_layer += 1

        #Append Final Layer
        if self.flattened:
            self.network.update({'logits': self.dense(self.network[next(reversed(self.network))], consts.CLASSIFICATIONS, 'logits')})
        else:
            self.network.update({'logits': self.dense(self.flatten(self.network[next(reversed(self.network))]), consts.CLASSIFICATIONS, 'logits')})
            self.flattened = True

        self.network_built = True

        #append loss function and then optimizer
        self.network.update({'loss': tf.reduce_mean(self.loss_function(labels = self.network['labels'], logits = self.network['logits']))})
        self.network.update({'optimizer': self.optimizer.minimize(self.network['loss'])})
        self.network.update({'correct_predictions': tf.equal(tf.argmax(self.network['logits'], 1), tf.argmax(self.network['labels'], 1))})
        self.network.update({'accuracy': tf.reduce_mean(tf.cast(self.network['correct_predictions'], tf.float32))})


        #train the model...includes validation
    def train(self):
        None if self.network_built else self.build_network() #Dynamically build the network if need be
        config = tf.ConfigProto()
        if self.gpu_mode == True: # set gpu configurations if specified
            config.gpu_options.allow_growth = True
        saver = tf.train.Saver() #ops to save the model
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer()) #initialize tf variables
            stop_count = 0 #variable that stores number of epochs model has trained without improving validation error
            # if we are starting with an existing model, take the current validation error
            if (self.existing_model):
                best_accuracy = self.validate(sess)
            else:
                best_accuracy = 0.0 #start with 0 accuracy

            while True: #we are going to find the number of epochs
                #if the number of training epochs is greater than the minimum number specified, and the model has not improved after a specified number of iterations, stop training
                if (stop_count > self.stop_threshold):
                    # release training memory here
                    self.dataset.free()
                    return

                total_train_accuracy = 0.0
                for step in tqdm(range(self.dataset.train_steps), desc = "Training Model " + str(self.id) + " - Epoch " + str(int(self.epochs+1))+" - Stop Count " + str(stop_count)):
                    train_photos, train_labels, train_filenames = self.dataset.next_train_batch() #get next training batch)
                    train_op, error, targets, outputs, acc = sess.run([self.network['optimizer'], self.network['loss'], self.network['labels'], self.network['logits'], self.network['accuracy']], feed_dict={self.network['inputs']: train_photos, self.network['labels']: train_labels}) #train and return predictions with target values

                #get current accuracy on validation set and train set
                total_train_accuracy += acc
                val_accuracy = self.validate(sess)
                print('Training accuracy:', total_train_accuracy/self.dataset.train_steps)
                print('Validation accuracy:', val_accuracy)

                if val_accuracy > best_accuracy: #if the accuracy on the val set is increasing
                    best_accuracy = val_accuracy
                    saver.save(sess, os.path.join(self.model_dir, str(self.id))) #save improved model
                    self.optimal_epochs = self.epochs #make the optimal number of epochs equal to the epoch with the best error
                    stop_count = 0 #model has improved, so stop_count is reset to zero
                else:
                    stop_count += 1 #if model has not improved, increment stop_count by one

                self.epochs += 1.

    #validation of model, similar to training but does not use the optimizer, returns mean squared error across the validation set.
    def validate(self, sess):
        total_accuracy = 0.0
        for step in tqdm(range(self.dataset.val_steps), desc = "Validating Model " + str(self.id) + "..."):
            val_photos, val_labels, val_filenames = self.dataset.next_val_batch()
            outputs, targets, error, accuracy = sess.run([self.network['logits'], self.network['labels'], self.network['loss'], self.network['accuracy']], feed_dict={self.network['inputs']: val_photos, self.network['labels']: val_labels}) #train and return predictions with target values
            total_accuracy += accuracy

        return total_accuracy / self.dataset.val_steps #return the avg error

    #restore the model and test... testing just returns testing filenames with their predictions
    def test(self):
        None if self.network_built else self.build_network() #Dynamically build the network if need be
        config = tf.ConfigProto()
        if self.gpu_mode == True: # set gpu configurations if specified
            config.gpu_options.allow_growth = True

        saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            saver.restore(sess, os.path.join(self.model_dir, str(self.id)))
            predictions = OrderedDict()
            for step in tqdm(range(self.dataset.test_steps), desc = "Testing Model " + str(self.id) + "..."):
                test_photos, test_filenames = self.dataset.next_test_batch() #get next training batch
                outputs = sess.run(self.network['logits'], feed_dict={self.network['inputs']: test_photos}) #train and return predictions with target values
                for i in range(len(test_filenames)):
                    predictions.update({test_filenames[i]: outputs[i]})
        return predictions

#-------------------------------------------------------------------------------
"""
    Unit Tests.
"""

if __name__ == '__main__':
    #Constants
    BATCH_SIZE = 50 #images per batch
    HDF5_FILE = str(sys.argv[1]) #path to hdf5 data file
    STORAGE_DIR = str(sys.argv[2])  #path to where we would like our model stored

    if str(sys.argv[3]).lower() == "gpu": #argument for whether or not 'cpu' or 'gpu' mode
        model = Model(HDF5_FILE, BATCH_SIZE, consts.TRAIN_PICS/2,       #sample gpu model that should fit on 3gb gpu
                                [32,64], [5,5], [2,2], [0.4],
                                [1024], tf.nn.softmax_cross_entropy_with_logits,
                                tf.train.AdamOptimizer(1e-4), 'CPCPDH',
                                STORAGE_DIR, gpu_mode=True)
    else:
        model = Model(HDF5_FILE, BATCH_SIZE, consts.TRAIN_PICS/2,       #sample model to run on cpu
                                [32,64], [5,5], [2,2], [0.4],
                                [1024], tf.nn.softmax_cross_entropy_with_logits,
                                tf.train.AdamOptimizer(1e-4), 'CPCPDH',
                                STORAGE_DIR, gpu_mode=False)


    model.train() #train the model
    predictions = model.test() #test the model and get the error
