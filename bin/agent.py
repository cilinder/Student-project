import sys
import math
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

        self.test = params.test

        # the agent_history_size parameter represents how many state frames we concatenate together before plugging them into the neural network
        self.input_data_size = params.input_data_size
        self.num_actions = params.num_actions or 5
        self.output_data_size = self.num_actions

        self.current_state = tf.placeholder(tf.float32, shape=[None, self.input_data_size])
        self.next_state = tf.placeholder(tf.float32, shape=[None, self.input_data_size])

        # Log data after every n steps
        self.log_data_every = params.log_data_every or 10

        # Record the data generated during learning or not
        self.record_learning_data = params.record_learning_data

        self.run_dir = params.run_dir

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

        # 
        # Select epsilon decay strategy
        self.epsilon_decay_strategy = params.epsilon_decay_strategy
        # Epsilon for e-greedy action selection strategy
        self.epsilon = params.epsilon
        self.start_epsilon = params.epsilon or 1
        self.end_epsilon = params.end_epsilon or 0.05
        self.epsilon_decay = params.epsilon_decay or 0.05
        self.learning_rate = params.learning_rate or 0.1

        self.horizon = tf.constant(0.99)
        self.reward_measured = tf.placeholder(tf.float32)

        self.y =  self.reward_measured + self.horizon * self.Qsa_frozen

        self.loss = tf.square(self.y - self.Qsa)


        #
        # Select optimization algorithm
        #
        if params.trainer == 'Gradient_descent':
            self.trainer = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif params.trainer == 'RMS_prop':
            self.trainer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif params.trainer == 'Adam':
            self.trainer = tf.train.AdamOptimizer(self.learning_rate)
        elif params.trainer == 'Momentum':
            self.trainer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9, use_nesterov=True)

        self.gvs = self.trainer.compute_gradients(self.loss)

        self.interesting_gradients = self.trainer.compute_gradients(self.loss, [W for W in self.W] + [b for b in self.b])

        #
        # Uncomment these lines to do gradient clipping
        #
        #self.capped_gvs = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in self.interesting_gradients]
        #self.train_step = self.trainer.apply_gradients(self.capped_gvs)
        self.train_step = self.trainer.apply_gradients(self.gvs)

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

        self.goals_reached = tf.placeholder(tf.int32)
        self.num_steps_in_run = tf.placeholder(tf.int32)
        self.total_reward = tf.placeholder(tf.int32)

        k = 0
        grad_summaries = []
        for grad, var in self.interesting_gradients:
            grad_summaries.append(tf.summary.scalar('gradient_'+var.name, tf.norm(grad)))
            k+=1

        W_summaries = []
        b_summaries = []
        for i in range(len(self.W)):
            W_summaries.append(tf.summary.histogram('W_'+str(i), self.W[i]))
            b_summaries.append(tf.summary.histogram('b_'+str(i), self.b[i]))

        self.running_summaries = tf.summary.merge([tf.summary.scalar('loss_function', self.loss), tf.summary.scalar('Qsa', self.Qsa)] + grad_summaries + W_summaries + b_summaries)
        self.run_summaries = tf.summary.merge([tf.summary.scalar('goals_reached', self.goals_reached), tf.summary.scalar('total_reward', self.total_reward),  tf.summary.scalar('num_steps', self.num_steps_in_run)])

        self.writer = tf.summary.FileWriter(self.run_dir, self.sess.graph)
        
        # Create a saver to save the model 
        self.saver = tf.train.Saver([W for W in self.W] + [b for b in self.b], max_to_keep = 10)

        if params.load_saved_agent:
            self.saver.restore(self.sess, params.logdir + '/' + params.load_saved_agent)
            for i in range(len(self.W_frozen)):
                self.W_frozen[i] = tf.Variable(self.W[i].initialized_value())
                self.b_frozen[i] = tf.Variable(self.b[i].initialized_value())


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

    def decay_epsilon(self):

        if self.epsilon_decay_strategy == 'linear':
            self.epsilon = max(self.epsilon - self.epsilon_decay, self.end_epsilon)


    def train_step(self, inputs, reward):

        self.sess.run([self.train_step], {self.inputs:inputs, self.reward_measured:reward})


    def learn_from_transition(self, current_state, action, reward, next_state, batch_size):

        # I don't know if I have to compute the interesting gradients sperately 
        #self.sess.run(self.interesting_gradients, {self.current_state:current_state, self.next_state:next_state, self.reward_measured:reward})

        step  = self.sess.run([self.train_step], feed_dict={self.current_state:current_state, self.next_state:next_state, self.reward_measured:reward})

        self.n_steps += batch_size

    def record_running_summaries(self, current_state, action, reward, next_state):
        
        summaries = self.sess.run(self.running_summaries, {self.current_state:current_state, self.next_state:next_state, self.reward_measured:reward})

        self.writer.add_summary(summaries, self.n_steps)

    def record_run_summaries(self, run_number,  goals_reached,  total_reward, num_steps_in_run):

        summaries = self.sess.run(self.run_summaries, {self.goals_reached:goals_reached, self.total_reward:total_reward, self.num_steps_in_run:num_steps_in_run})

        self.writer.add_summary(summaries, run_number)


    def update_frozen_network(self):

        self.sess.run(self.update_W_frozen)
        self.sess.run(self.update_b_frozen)


    def save_network(self, global_step):

        self.saver.save(self.sess, self.run_dir + '/Q-network', global_step=global_step)






