import sys
import tensorflow as tf
import numpy as np

from tensorflow.python import debug as tf_debug

import matplotlib.pyplot as plt


class Agent:

    class SelectionMechanism:
        Greedy, EpsilonGreedy = range(2)


    def __init__(self, params):

        #########################################################
        #                                                       #
        #   Initialize all the parameters                       #
        #                                                       #
        #########################################################

        self.n_steps = 0

        # the agent_history_size parameter represents how many state frames we concatenate together before plugging them into the neural network
        self.input_data_size = params.input_data_size * params.agent_history_size
        self.num_actions = params.num_actions or 5
        self.output_data_size = self.num_actions

        self.current_state = tf.placeholder(tf.float32, shape=[None, self.input_data_size])
        self.next_state = tf.placeholder(tf.float32, shape=[None, self.input_data_size])

        # Log data after every n steps
        self.log_data_every = params.log_data_every or 10

        self.run_dir = params.run_dir
        self.run = params.run

        #############################################################
        #                                                           #
        # Create the 2 networks needed for reinforcement learning   #
        #                                                           #
        #############################################################

        self.num_hidden_layers = params.num_hidden_layers or 1
        self.hidden_layer_size = params.hidden_layer_size or 10
        self.hidden_layer_sizes = [self.hidden_layer_size for _ in range(self.num_hidden_layers)]

        ##
        ## Select wanted activation function
        ##
        #self.activation = tf.sigmoid
        #self.activation = tf.tanh
        self.activation = tf.nn.relu
        #self.activation = tf.nn.relu6
        #self.activation = tf.nn.elu

        self.createQNets()

        # Epsilon for e-greedy action selection strategy
        self.epsilon = params.epsilon
        #self.start_epsilon = params.epsilon or 1
        #self.end_epsilon = 0.05
        #self.decay_epsilon_by = params.decay_epsilon_by or 0.1
        #self.decay_epsilon_every = params.decay_epsilon_every or 10000
        self.learning_rate = params.learning_rate or 0.1

        self.horizon = tf.constant(0.99)
        self.reward_measured = tf.placeholder(tf.float32)

        self.y =  self.reward_measured + self.horizon * self.Qsa_frozen

        self.loss = tf.square(self.y - self.Qsa)


        #
        # Select optimization algorithm
        #
        self.trainer = tf.train.GradientDescentOptimizer(self.learning_rate)
        #self.trainer = tf.train.RMSPropOptimizer(self.learning_rate)
        #self.trainer = tf.train.AdamOptimizer(self.learning_rate)
        #self.trainer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9, use_nesterov=True)

        self.gvs = self.trainer.compute_gradients(self.loss)

        self.interesting_gradients = self.trainer.compute_gradients(self.loss, [W for W in self.W] + [b for b in self.b])

        #
        # Uncomment these lines to do gradient clipping
        #
        self.capped_gvs = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in self.interesting_gradients]
        self.train_step = self.trainer.apply_gradients(self.capped_gvs)
        #self.train_step = self.trainer.apply_gradients(self.gvs)

        self.sess = tf.Session()

        self.debug_mode = params.debug_mode or False

        if self.debug_mode:
            self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
            self.sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)


        # Create operation to update the frozen network
        self.update_W_frozen = []
        self.update_b_frozen = []
        for i in range(len(self.W_frozen)):
            self.update_W_frozen.append(self.W_frozen[i].assign(self.W[i]))
            self.update_b_frozen.append(self.b_frozen[i].assign(self.b[i]))


        # Add the summary operations
        tf.summary.scalar('loss_function', self.loss)
        tf.summary.scalar('Qsa', self.Qsa)

        k = 0
        for grad, var in self.interesting_gradients:
            tf.summary.scalar('gradient_'+var.name, tf.norm(grad))
            k+=1

        for i in range(len(self.W)):
            tf.summary.histogram('W_'+str(i), self.W[i])
            tf.summary.histogram('b_'+str(i), self.b[i])

        #tf.summary.histogram('learning updates', self.train_step)

        self.merged_summaries = tf.summary.merge_all()

        self.writer = tf.summary.FileWriter(self.run_dir, self.sess.graph)
        
        # Create a saver to save the model 
        self.saver = tf.train.Saver([W for W in self.W] + [b for b in self.b], max_to_keep = 10)

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def createQNets(self):

        dimensions = [self.input_data_size] + self.hidden_layer_sizes + [self.output_data_size]

        # Create the first neural network weights and biases
        self.W = []
        self.W_frozen = []
        for i in range(len(dimensions) - 1):
            W_init_tensor = self.initialization(dimensions[i], dimensions[i+1])
            self.W.append(tf.Variable(W_init_tensor, dtype=tf.float32, name='W_'+str(i)))
            self.W_frozen.append(tf.Variable(self.W[i].initialized_value(), dtype=tf.float32, name='W_frozen_'+str(i)))

        self.b = []
        self.b_frozen = []
        for i in range(1, len(dimensions)):
            b_init_tensor = self.initialization(dimensions[i])
            self.b.append(tf.Variable(b_init_tensor, dtype=tf.float32, name='b_'+str(i)))
            self.b_frozen.append(tf.Variable(self.b[i-1].initialized_value(), dtype=tf.float32, name='b_frozen_'+str(i)))


        # First layer computation depends on input
        self.hidden_layers = [self.activation(tf.matmul(self.current_state, self.W[0]) + self.b[0])]
        self.hidden_layers_frozen = [self.activation(tf.matmul(self.next_state, self.W_frozen[0]) + self.b_frozen[0])]

        for i in range(1, self.num_hidden_layers):
            self.hidden_layers.append(self.activation(tf.matmul(self.hidden_layers[i-1], self.W[i]) + self.b[i]))
            self.hidden_layers_frozen.append(self.activation(tf.matmul(self.hidden_layers_frozen[i-1], self.W_frozen[i]) + self.b_frozen[i]))

        self.Qs = tf.matmul(self.hidden_layers[self.num_hidden_layers-1], self.W[self.num_hidden_layers]) + self.b[self.num_hidden_layers]
        self.Qsa = tf.reduce_max(self.Qs)

        self.Qs_frozen = tf.matmul(self.hidden_layers_frozen[self.num_hidden_layers-1], self.W[self.num_hidden_layers]) + self.b_frozen[self.num_hidden_layers]
        self.Qsa_frozen = tf.reduce_max(self.Qs_frozen)

        self.predict_action = tf.argmax(self.Qs, 1)


    def initialization(self, dim_input, dim_output=None):

        if dim_output == None:
            # Only vector of size dim_input expected
            return tf.random_normal([dim_input], stddev=0.1)

        return tf.random_normal([dim_input, dim_output], mean=1, stddev=0.1)


    def select_greedy(self, state):

        action = np.zeros(5)
        selected_action = self.sess.run([self.predict_action], feed_dict={self.current_state:state})
        action[selected_action] = 1

        return action


    def select_epsilon_greedy(self, state):

        action = np.zeros(5, dtype=np.int_)

        if np.random.uniform() > self.epsilon:
            selected_action = self.sess.run([self.predict_action], feed_dict={self.current_state:state})
            action[selected_action] = 1
        else:
            action[np.random.randint(5)] = 1

        return action



    def train_step(self, inputs, reward):

        self.sess.run([self.train_step], {self.inputs:inputs, self.reward_measured:reward})


    def learn_from_transition(self, current_state, action, reward, next_state, batch_size):

        # I don't know if I have to compute the interesting gradients sperately 
        #self.sess.run(self.interesting_gradients, {self.current_state:current_state, self.next_state:next_state, self.reward_measured:reward})

        step  = self.sess.run([self.train_step], feed_dict={self.current_state:current_state, self.next_state:next_state, self.reward_measured:reward})

        summaries = self.sess.run(self.merged_summaries, {self.current_state:current_state[0].reshape(1,-1), self.next_state:next_state[0].reshape(1,-1), self.reward_measured:reward[0]})

        if self.n_steps % self.log_data_every == 0:
            self.writer.add_summary(summaries, self.n_steps)

        self.n_steps += batch_size


    def update_frozen_network(self):

        self.sess.run(self.update_W_frozen)
        self.sess.run(self.update_b_frozen)


    def save_network(self, global_step):

        self.saver.save(self.sess, self.run_dir + '/Q-network', global_step=global_step)






