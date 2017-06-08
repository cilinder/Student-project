import sys
import argparse

import numpy as np
import tensorflow as tf

import pygame
from pygame.locals import *

import enviroment
from enviroment import Enviroment

from agent import Agent
from replayMemory import ReplayMemory

import matplotlib.pyplot as plt

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


def main(argv):

    global frame_by_frame_mode
    global next_frame

    FLAGS.run_dir = FLAGS.logdir + '/run_' + str(time.time())

    env = Enviroment(argv, FLAGS.display_world)

    input_dim = env.level.boat.number_of_rays + 2

    FLAGS.input_data_size = input_dim
    FLAGS.num_actions = 5

    tf.logging.set_verbosity(tf.logging.INFO)

    agent = Agent(FLAGS)

    memory = ReplayMemory(FLAGS)

    num_games = FLAGS.num_games

    accumulated_rewards = []
    actions = np.zeros(FLAGS.num_actions)

    # Total ammount of time steps, frozen network is updated every C iterations
    update_network_every = 100
    total_steps = 1

    plt.ion()
    fig = plt.figure(1)

    for game in range(num_games):

        # The total reward for this game, we want to maximize this in principle
        total_reward = 0

        # Reset the game before the start of every new instance
        env.resetGame()

        time_step = 0
        current_state = env.currentState()

        #print('Qs, Qs_frozen: ', agent.sess.run([agent.Qs, agent.Qs_frozen], feed_dict={agent.current_state:current_state, agent.next_state:current_state}))

        while not env.gameHasEnded():

            checkEventQueue(env)

            if frame_by_frame_mode and not next_frame:
                continue
            elif frame_by_frame_mode and next_frame:
                print('Next frame')
                next_frame = False

            if env.manual_control:
                keys = pygame.key.get_pressed()
                action = handleKeyboardInput(keys)
            else:
                action = agent.select_epsilon_greedy(current_state)


            actions = actions + action
            
            if total_steps % 1000 == 0:
                ax = fig.gca()
                plt.cla()
                actions_bar = actions / np.linalg.norm(actions)
                plt.title('actions distribution')
                plt.bar(range(len(actions)), actions_bar)
                ax.set_xticklabels(('Do noting', 'Do noting', 'Power', 'Break', 'Left', 'Right'))
                plt.pause(0.000001)
            
            env.nextState(action)

            next_state = env.currentState()

            reward = env.getReward()
            total_reward += reward

            activations = agent.sess.run([layer for layer in agent.hidden_layers], feed_dict={agent.current_state:current_state})

            """
            plt.title('hidden layer activations')
            figure=1
            for hidden_layer in activations:
                plt.figure(figure)
                plt.bar(range(len(hidden_layer[0])), hidden_layer[0])
                figure+=1
                plt.pause(0.1)
            """

            W_0, W_1 = agent.sess.run([agent.W[0], agent.W[1]])

            #print('norm of weights: ', np.linalg.norm(W_0), np.linalg.norm(W_1))

            grads, Qs, Qs_frozen, y, loss = agent.sess.run([agent.interesting_gradients, agent.Qs, agent.Qs_frozen, agent.y, agent.loss], {agent.current_state:current_state, agent.next_state:next_state, agent.reward_measured:reward})

            agent.learn_from_transition(action, current_state, next_state, reward)

            if total_steps % update_network_every == 0:
                agent.update_frozen_network()

            memory.save_transition(action, current_state, next_state, reward)

            current_state = next_state

            total_steps += 1
            time_step+=1


        accumulated_rewards.append(total_reward)
        print('Game number ', game, ' has ended')
        print('Boat has come to goal: ', env.level.boat.goal_reached)
        print('Total reward was: ', total_reward)

        agent.save_network(game)


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='/home/ros/Student_project/train_data', help='Directory for saving data')
    parser.add_argument('--run', type=str, default='1', help="The run number")
    parser.add_argument('--num_games', type=int, default=5, help="Number of games for the simulation to run")
    parser.add_argument('--debug_mode', dest='debug_mode', action='store_true', help='Toggle debug mode on/off')
    parser.add_argument('--no-debug_mode', dest='debug_mode', action='store_false', help='Toggle debug mode on/off')
    parser.add_argument('--log_data_every', type=int, default=100, help='Log a data point every n steps')
    parser.add_argument('--num_hidden_layers', type=int, default=1, help='The number of hidden layers for the neural network')
    parser.add_argument('--hidden_layer_size', type=int, default=10, help='The size of the network hidden layers (all layers same size atm)')
    parser.add_argument('--epsilon', type=float, default=0.05, help='The probability of selecting a random action')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='The learnig rate')
    parser.add_argument('--display_world', dest='display_world', action='store_true', help='Use this feature if you want the simulation to be displayed')
    parser.add_argument('--no-display_world', dest='display_world', action='store_false', help='Use this feature if you want to run the simulation without displaying it')
    parser.set_defaults(display_world=True)
    parser.set_defaults(debug_mode=False)
    FLAGS, unparsed = parser.parse_known_args()

    pygame.init()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

    pygame.quit()

    print("Done!")

