import argparse
import sys
import os
import pickle

import numpy as np
import tensorflow as tf

import pygame
from pygame.locals import *

import enviroment
from enviroment import Enviroment

from agent import Agent
from replayMemory import ReplayMemory

import matplotlib.pyplot as plt

from datetime import datetime
import time

FLAGS = None
frame_by_frame_mode = False
next_frame = True

def checkEventQueue(enviroment):
    global frame_by_frame_mode
    global next_frame

    # Check the event que
    for event in pygame.event.get():
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            # The user closed the window or pressed escape
            enviroment.running = False
        if event.type == KEYDOWN:
            if event.key == K_g:
                enviroment.display_grid = not enviroment.display_grid
            if event.key == K_PLUS:
                m = min(enviroment.SCREEN_HEIGHT,enviroment.SCREEN_WIDTH) / enviroment.PPM
                enviroment.x_spacing = min(m, enviroment.x_spacing + 0.5)
                enviroment.y_spacing = min(m, enviroment.y_spacing + 0.5)
            if event.key == K_MINUS:
                enviroment.x_spacing = max(0.5, enviroment.x_spacing - 0.5)
                enviroment.y_spacing = max(0.5, enviroment.y_spacing - 0.5)
            if event.key == K_p:
                enviroment.pause = not enviroment.pause
            if event.key == K_h:
                enviroment.display_hud = not enviroment.display_hud
            if event.key == K_r:
                enviroment.resetGame()
            if event.key == K_o:
                enviroment.display_highscores = not enviroment.display_highscores
            if event.key == K_b:
                enviroment.boat_vision_on = not enviroment.boat_vision_on
            if event.key == K_m:
                enviroment.manual_control = not enviroment.manual_control
            if event.key == K_f:
                frame_by_frame_mode = not frame_by_frame_mode
            if not next_frame and event.key == K_n:
                next_frame = True



# Also want to move this to the learner class
def handleKeyboardInput(keys):

    if keys[K_w]:
        action = enviroment.INCREASE_POWER
    elif keys[K_s]:
        action = enviroment.DECREASE_POWER
    elif keys[K_a]:
        action = enviroment.STEER_LEFT
    elif keys[K_d]:
        action = enviroment.STEER_RIGHT
    else:
        action = enviroment.DO_NOTHING

    return action

def display_values(values, labels=None):

    #fig, ax = plt.subplots()

    #values = values / np.linalg.norm(values)

    index = np.arange(len(values))

    #bar_width = ?

    rect = plt.bar(index, values, color='b', label='values')

    if labels:
        plt.xticks(index, labels)

    plt.pause(0.00001)


def main(argv):
    global frame_by_frame_mode
    global next_frame


    # Create the enviroment
    env = Enviroment(FLAGS)

    # Append the data based on the enviroment used
    FLAGS.input_data_size = env.state_size
    FLAGS.num_actions = 5

    # Create the agent used to navigate in the world
    agent = Agent(FLAGS)

    # The replay memory
    memory = ReplayMemory(FLAGS)

    ########################
    # Setup some variables #
    ########################
    accumulated_rewards = []
    actions = np.zeros(FLAGS.num_actions)
    update_network_every = 100
    total_steps = 0
    goals_reached = 0
    ########################

    # Used for live plot updating in matplotlib
    plt.ion()


    learning_start_time = time.clock()

    for game in range(FLAGS.num_games):

        #####################################
        # Initialize all the run variables  #
        #####################################

        # The total reward for this game, we want to maximize this in principle
        total_reward = 0
        time_step = 0
        # Reset the game before the start of every new instance
        env.resetGame()
        current_state = env.currentState()
        actions_distr = np.zeros(5)


        while not env.gameHasEnded():

            checkEventQueue(env)

            if frame_by_frame_mode and not next_frame:
                continue
            elif frame_by_frame_mode and next_frame:
                print('Next frame')
                next_frame = False


            total_steps += 1
            time_step += 1

            if env.manual_control:
                keys = pygame.key.get_pressed()
                action = handleKeyboardInput(keys)
            else:
                # Only start predicting actions after a certain amount of time, before that select random actions and only fill the memory
                if total_steps < FLAGS.replay_start_size:
                    action = np.zeros(FLAGS.num_actions)
                    action[np.random.randint(FLAGS.num_actions)] = 1
                else:
                    action = agent.select_epsilon_greedy(current_state)

            # Advance the enviroment by one step based on the selected action
            env.nextState(action)

            if FLAGS.test:
                continue

            # If test flag in turned on this is where the loop ends 
            ##############################################################################
            ##############################################################################

            # Update the action distribution statistics
            actions_distr = actions_distr + action

            if FLAGS.display_action_distribution and total_steps % 100 == 0:
                display_values(actions_distr)

            # Update the accumulated state vector (n last states concatenated into a single vector, we feed this into the neural network)
            next_state = env.currentState()

            # Get the reward for this state and acumulate the total reward for this run
            reward = env.getReward()
            total_reward += reward

            # Save the transition into memory
            memory.save_transition(current_state, action, reward, next_state)

            # If we are only collecting experience atm we don't want to perform learning just yet
            if total_steps <= FLAGS.replay_start_size:
                current_state = next_state
                continue
            elif total_steps == FLAGS.replay_start_size:
                print("Ending random collection of data")

            #################################
            """
            if total_steps % 1000 == 0:
                activations = agent.sess.run([layer for layer in agent.hidden_layers], feed_dict={agent.current_state:current_state})
                plt.clf()
                for i in range(1, len(activations)+1):
                    plt.subplot(1, len(activations), i)
                    plt.hist(activations[i-1].flatten(), bins=10)
                    plt.pause(0.000001)
            """
            ###################################

            # Sample of minibatch of experiences
            minibatch = memory.sample(FLAGS.minibatch_size)

            # Unpack the minibatch
            current_states, actions, rewards, next_states, batch_size = minibatch

            # Learn from the sampled transitions
            agent.learn_from_transition(current_states, actions, rewards, next_states, batch_size)

            # Decay the epsilon (for epsilon-greedy strategy selection)
            if total_steps % FLAGS.decay_epsilon_every == 0:
                agent.decay_epsilon()

            # Update the frozen network every n steps
            if total_steps % update_network_every == 0:
                agent.update_frozen_network()

            # Record the data every n time steps if recording activated
            if FLAGS.record_learning_data and (total_steps % FLAGS.log_data_every == 0):
                agent.record_running_summaries(current_state, action, reward, next_state)

            current_state = next_state


        if env.level.boat.goal_reached:
            goals_reached += 1

        agent.record_run_summaries(game,  goals_reached,  total_reward, time_step)

        print('Game number ', game, ' has ended in ', time_step, ' steps.')
        print('Boat has come to goal: ', env.level.boat.goal_reached)
        print('Total reward was: ', total_reward)
        print('Average game time: ', (time.clock() - learning_start_time) / (game+1))

        if FLAGS.save_agent:
            agent.save_network(game)



if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--player_name', type=str, default='Jure', help='Name of the human (or non-human) player')
    parser.add_argument('--level', type=str, default='1', help='The name of the level to load')
    parser.add_argument('--manual_control', dest='manual_control', action='store_true', help='Use this option if you want to steer the boat yourself')
    parser.add_argument('--logdir', type=str, default='/home/ros/Student_project/train_data', help='Directory for saving data')
    parser.add_argument('--num_games', type=int, default=1, help="Number of games for the simulation to run")
    parser.add_argument('--debug_mode', dest='debug_mode', action='store_true', help='Toggle debug mode on/off')
    parser.add_argument('--no-debug_mode', dest='debug_mode', action='store_false', help='Toggle debug mode on/off')
    parser.add_argument('--log_data_every', type=int, default=1000, help='Log a data point every n steps')
    parser.add_argument('--num_hidden_layers', type=int, default=1, help='The number of hidden layers for the neural network')
    parser.add_argument('--hidden_layer_size', type=int, default=10, help='The size of the network hidden layers (all layers same size atm)')
    parser.add_argument('--epsilon', type=float, default=1, help='The probability of selecting a random action')
    parser.add_argument('--end_epsilon', type=float, default=0.05, help='The minimum epsilon decayed to, with the selected epsilon decay')
    parser.add_argument('--epsilon_decay', type=float, default=0.05, help='How much to decrease the epsilon by with linear decay')
    parser.add_argument('--epsilon_decay_strategy', type=str, default='linear', help='The was to decay the epsilon during learning, options: [linear]')
    parser.add_argument('--decay_epsilon_every', type=int, default=50000, help='On how many steps to decay the epsilon')
    parser.add_argument('--trainer', type=str, default='Adam', help='Select the optimization algorithm, options: [Gradient_descent, RMS_prop, Adam, Momentum]')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='The learnig rate')
    parser.add_argument('--display_world', dest='display_world', action='store_true', help='Use this feature if you want the simulation to be displayed')
    parser.add_argument('--no-display_world', dest='display_world', action='store_false', help='Use this feature if you want to run the simulation without displaying it')
    parser.add_argument('--minibatch_size', type=int, default=1, help='The size of the minibatch to sample from memory to learn from')
    parser.add_argument('--replay_start_size', type=int, default=100, help='A uniform policy is run for this number of steps before learning starts and the resulting experience is used to populate the replay memory')
    parser.add_argument('--max_time_ticks', type=int, default=5000, help='The maximum number of ticks to run the simulation before reseting. Prevents getting stuck too much')
    parser.add_argument('--memory_max_size', type=int, default=100000, help='The number of last frames seen to be kept in memory')
    parser.add_argument('--display_action_distribution', dest='display_action_distribution', action='store_true', help='Use this if you want a live histogram of selected actions')
    parser.add_argument('--ticks_per_second', type=int, default=10, help='Sets the number of time the agent gets to make a decision every second')
    parser.add_argument('--record_learning_data', dest='record_learning_data', action='store_true', help='Toggle to save/not-save the data created while learning')
    parser.add_argument('--save_agent', dest='save_agent', action='store_true', help='Toggle to save the agent (network parameters)')
    parser.add_argument('--no-record_learning_data', dest='record_learning_data', action='store_false', help='Toggle to save/not-save the data created while learning')
    parser.add_argument('--no-save_agent', dest='save_agent', action='store_false', help='Toggle to save the agent (network parameters)')
    parser.add_argument('--load_saved_agent', type=str, help='Select the folder to load the saved agent from')
    parser.add_argument('--test', action='store_true', help='Use this flag if you want to test an agent without learning')
    parser.set_defaults(display_world=True)
    parser.set_defaults(debug_mode=False)
    parser.set_defaults(manual_control=False)
    parser.set_defaults(display_action_distribution=False)
    parser.set_defaults(record_learning_data=True)
    parser.set_defaults(save_agent=True)
    FLAGS, unparsed = parser.parse_known_args()

    # Create the name of the run directory based on the current datetime
    FLAGS.run_dir = FLAGS.logdir + '/run_' + str(datetime.now())

    if FLAGS.load_saved_agent:
        with open(FLAGS.logdir + '/' + FLAGS.load_saved_agent + '/.params.pickle') as f:
            LOADED_FLAGS = pickle.load(f)

            FLAGS.num_hidden_layers = LOADED_FLAGS.num_hidden_layers
            FLAGS.hidden_layer_size = LOADED_FLAGS.hidden_layer_size
            FLAGS.minibatch_size = LOADED_FLAGS.minibatch_size
            FLAGS.action_repeat = LOADED_FLAGS.action_repeat

    # Save the run parameters to a file
    # First create the folder is not created already
    if not os.path.exists(FLAGS.run_dir):
        os.makedirs(FLAGS.run_dir)
    if FLAGS.record_learning_data:
        with open(FLAGS.run_dir + '/.params.pickle', 'w') as f:
            pickle.dump(FLAGS, f)

    pygame.init()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

    pygame.quit()

    print("Done!")

