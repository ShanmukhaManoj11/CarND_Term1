import tensorflow as tf
from tensorflow.contrib.layers import flatten
import math
from sklearn.utils import shuffle

def lenet(x,keep_prob):
    mu=0
    sigma=0.1
    n_classes=43
    # layer 1 - convolutional, input = 32x32x3, output = 28x28x16
        # relu activation
        # maxpooling, input = 28x28x16, output = 14x14x16
    # layer 2 - convolutional, input = 14x14x16, output = 10x10x32
        # relu activation
        # maxpooling, input = 10x10x32, output = 5x5x32
    # flatten, input = 5x5x32, output = 800
    # fully connected, input = 800, output = 200
        # relu activation
    # fully connected, input = 200, output = 100
        # relu activation
    # fully connected, input = 100, output = 43
    weights={
        'wc1':tf.Variable(tf.truncated_normal((5,5,3,16),mean=mu,stddev=sigma)),
        'wc2':tf.Variable(tf.truncated_normal((5,5,16,32),mean=mu,stddev=sigma)),
        'wf1':tf.Variable(tf.truncated_normal((800,200),mean=mu,stddev=sigma)),
        'wf2':tf.Variable(tf.truncated_normal((200,100),mean=mu,stddev=sigma)),
        'wf3':tf.Variable(tf.truncated_normal((100,n_classes),mean=mu,stddev=sigma))
    }
    biases={
        'bc1':tf.Variable(tf.zeros(16)),
        'bc2':tf.Variable(tf.zeros(32)),
        'bf1':tf.Variable(tf.zeros(200)),
        'bf2':tf.Variable(tf.zeros(100)),
        'bf3':tf.Variable(tf.zeros(n_classes))
    }
    # layer 1
    x=tf.nn.conv2d(x,weights['wc1'],strides=[1,1,1,1],padding='VALID')
    x=tf.nn.bias_add(x,biases['bc1'])
    x=tf.nn.relu(x)
    x=tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    # layer 2
    x=tf.nn.conv2d(x,weights['wc2'],strides=[1,1,1,1],padding='VALID')
    x=tf.nn.bias_add(x,biases['bc2'])
    x=tf.nn.relu(x)
    x=tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    # flatten
    x=flatten(x)
    x=tf.nn.dropout(x,keep_prob)
    # layer 3
    x=tf.add(tf.matmul(x,weights['wf1']),biases['bf1'])
    x=tf.nn.relu(x)
    x=tf.nn.dropout(x,keep_prob)
    # layer 4
    x=tf.add(tf.matmul(x,weights['wf2']),biases['bf2'])
    x=tf.nn.relu(x)
    x=tf.nn.dropout(x,keep_prob)
    # layer 5
    x=tf.add(tf.matmul(x,weights['wf3']),biases['bf3'])
    return x

# architecture similar to Sermanet and LeCun's paper (published baseline model)
def NN(x,keep_prob):
    mu=0
    sigma=0.1
    n_classes=43
    # layer 1 - convolutional (5x5 filters/stride: 1), input = 32x32x3, output = 28x28x32 
        # relu activation
    # maxpooling 1 - (2x2/stride: 2), input = 28x28x32, output = 14x14x32
    # layer 2a - convolutional (5x5 filters/stride: 1), input = 14x14x32, output = 10x10x64
        # relu activation
    # maxpooling 2a - (2x2/stride: 2), input = 10x10x64, output = 5x5x64 
    # layer 3a - convolutional (5x5 filters/stride: 1), input = 5x5x64, output = 1x1x400
        # relu activation
    # concatenate flattened layer 3a output and maxpooling 1 output
        # 400 + 6272
    # fully connected layer 1, input = 6672, output = 1024
        # relu activation
    # fully connected layer 2, input = 1024, output = 43(n_classes)
    weights={
        'wc1':tf.Variable(tf.truncated_normal((5,5,3,32),mean=mu,stddev=sigma)),
        'wc2a':tf.Variable(tf.truncated_normal((5,5,32,64),mean=mu,stddev=sigma)),
        'wc3a':tf.Variable(tf.truncated_normal((5,5,64,400),mean=mu,stddev=sigma)),
        'wf1':tf.Variable(tf.truncated_normal((6672,1024),mean=mu,stddev=sigma)),
        'wf2':tf.Variable(tf.truncated_normal((1024,n_classes),mean=mu,stddev=sigma))
    }
    biases={
        'bc1':tf.Variable(tf.zeros(32)),
        'bc2a':tf.Variable(tf.zeros(64)),
        'bc3a':tf.Variable(tf.zeros(400)),
        'bf1':tf.Variable(tf.zeros(1024)),
        'bf2':tf.Variable(tf.zeros(n_classes))
    }
    # conv layer 1
    x=tf.nn.conv2d(x,weights['wc1'],strides=[1,1,1,1],padding='VALID')
    x=tf.nn.bias_add(x,biases['bc1'])
    x=tf.nn.relu(x)
    # maxpool 1
    maxpool1_out=x=tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    # conv layer 2a
    x=tf.nn.conv2d(x,weights['wc2a'],strides=[1,1,1,1],padding='VALID')
    x=tf.nn.bias_add(x,biases['bc2a'])
    x=tf.nn.relu(x)
    # maxpool 2a
    x=tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    # conv layer 3a
    x=tf.nn.conv2d(x,weights['wc3a'],strides=[1,1,1,1],padding='VALID')
    x=tf.nn.bias_add(x,biases['bc3a'])
    x=tf.nn.relu(x)
    #concatenate
    x=tf.concat((flatten(x),flatten(maxpool1_out)),axis=1)
    x=tf.nn.dropout(x,keep_prob)
    # fully connected layer 1
    x=tf.add(tf.matmul(x,weights['wf1']),biases['bf1'])
    x=tf.nn.relu(x)
    x=tf.nn.dropout(x,keep_prob)
    # fully connected layer 2
    x=tf.add(tf.matmul(x,weights['wf2']),biases['bf2'])
    return x

class Model(object):
    
    def __init__(self,architecture,model_name='model'):
        self.model_name=model_name
        self.x=tf.placeholder(tf.float32,(None,32,32,3),name='input')
        self.y=tf.placeholder(tf.int32,(None),name='ground_truth')
        n_classes=43
        self.one_hot_y=tf.one_hot(self.y,n_classes)
        self.keep_prob=tf.placeholder(tf.float32,name='keep_prob')
        self.learning_rate=tf.placeholder(tf.float32,name='learning_rate')
        self.logits=architecture(self.x,self.keep_prob)
        
        self.cross_entropy=tf.nn.softmax_cross_entropy_with_logits(labels=self.one_hot_y,logits=self.logits,name='cross_entropy')
        self.loss_operation=tf.reduce_mean(self.cross_entropy,name='loss_operation')
        self.optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate,name='optimizer')
        self.training_operation=self.optimizer.minimize(self.loss_operation,name='training_operation')

        self.correct_predictions=tf.equal(tf.argmax(self.logits,1),tf.argmax(self.one_hot_y,1),name='correct_predictions')
        self.accuracy_operation=tf.reduce_mean(tf.cast(self.correct_predictions,tf.float32,name='accuracy_operation'))

        self.sess=tf.get_default_session()
        
    def evaluate(self,X_data,y_data,BATCH_SIZE=128,load_model=False):
        if load_model:
            self.sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
            tf.train.Saver().restore(self.sess,'./'+self.model_name+'.ckpt')
        n_examples=len(X_data)
        n_batches=math.ceil(n_examples/BATCH_SIZE)
        t_accuracy=0
        offset=0
        for batch in range(n_batches):
            X_=X_data[offset:offset+BATCH_SIZE]
            y_=y_data[offset:offset+BATCH_SIZE]
            offset+=BATCH_SIZE
            accuracy=self.sess.run(self.accuracy_operation,feed_dict={self.x:X_,self.y:y_,self.keep_prob:1.0})
            t_accuracy+=accuracy*len(X_)
        return t_accuracy/n_examples

    def train(self,X_train,y_train,X_valid,y_valid,RESUME_TRAINING=False,LEARNING_RATE=0.001,EPOCHS=20,BATCH_SIZE=128,KEEP_PROB=1.0):
        saver2save=tf.train.Saver()    
        self.sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
        if RESUME_TRAINING:
            tf.train.Saver().restore(self.sess,'./'+self.model_name+'.ckpt')
            
        n_examples=len(X_train)
        n_batches=math.ceil(n_examples/BATCH_SIZE)
        for epoch in range(EPOCHS):
            X_train,y_train=shuffle(X_train,y_train)
            offset=0
            training_loss=0
            for batch in range(n_batches):
                X_=X_train[offset:offset+BATCH_SIZE]
                y_=y_train[offset:offset+BATCH_SIZE]
                offset+=BATCH_SIZE
                _,loss=self.sess.run([self.training_operation,self.loss_operation],
                                     feed_dict={self.x:X_,self.y:y_,self.keep_prob:KEEP_PROB,self.learning_rate:LEARNING_RATE})
                training_loss+=loss*len(X_)
            training_loss=training_loss/n_examples
            validation_accuracy=self.evaluate(X_valid,y_valid)
            print('Epoch {} ...'.format(epoch+1))
            print('training loss = {:.4f}, validation accuracy = {:.4f}'.format(training_loss,validation_accuracy))
            if (epoch+1)%5==0:
                saver2save.save(self.sess,'./'+self.model_name+'.ckpt')
                print("Model Saved.")
        if EPOCHS%5!=0:
            saver2save.save(self.sess,'./'+self.model_name+'.ckpt')
            print("Model Saved.")
        
        
        
        
    
