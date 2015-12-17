"""
The NeuralCoach class wraps a deep Q-network for training and testing
in the Arcade learning environment.

Author: Nathan Sprague

"""

import os
import cPickle
import time
import logging

import numpy as np

import ale_data_set

import sys

class NeuralCoach(object):

    def __init__(self, q_network, dataset, epsilon, random_state):

        self.network = q_network
        self.epsilon = epsilon
        self.dataset = dataset
        self.random_state = random_state

        self.phi_length = self.network.num_frames
        self.num_actions = self.network.num_actions

        logging.info( "NeuralCoach: actions %d, phi %d" % ( self.num_actions, self.phi_length, ) )

    def _choose_action( self, dataset, epsilon, observation ) :
        if ( self.random_state.rand( ) < epsilon ) or ( self.step_counter_ < self.phi_length ) :
            return self.random_state.randint( 0, self.num_actions )
        return self.network.choose_action( dataset.phi( observation ) )

    def start_epoch( self, epoch, testing ) :
        assert not ( hasattr( self, 'epoch_in_progress_' ) and self.epoch_in_progress_ )
        self.epoch_in_progress_ = True

        logging.info( "coaching epoch %d: START" % ( epoch, ) )

        self.current_epoch_, self.epoch_start_ = epoch, time.time( )
        self.epoch_reward_, self.episode_counter_ = 0, 0

    def start_episode( self, observation ):
        assert self.epoch_in_progress_
        assert not ( hasattr( self, 'episode_in_progress_' ) and self.episode_in_progress_ )
        self.episode_in_progress_ = True

        logging.info( "coaching episode %d: START %f" % ( self.episode_counter_, self.epsilon ) )

        self.step_counter_, self.episode_reward_ = 0, 0
        self.current_episode_ = self.episode_counter_
        self.episode_start_ = time.time( )

        action = self._choose_action( self.dataset, 1.0, observation )
        self.last_action_, self.last_observation_ = action, observation
        return action

    def step( self, reward, observation ) :
        assert self.episode_in_progress_

        self.dataset.add_sample( self.last_observation_, self.last_action_,
                                          np.clip( reward, -1, 1 ), terminal = False )

        self.step_counter_ += 1
        self.episode_reward_ += reward

        action = self._choose_action( self.dataset, self.epsilon, observation )

        self.last_action_, self.last_observation_ = action, observation
        return action

    def end_episode(self, reward, terminal = True ):
        assert self.epoch_in_progress_ and self.episode_in_progress_
        self.episode_in_progress_ = False

        self.dataset.add_sample( self.last_observation_, self.last_action_,
                                 np.clip( reward, -1, 1 ), terminal = True )
        self.step_counter_ += 1
        self.episode_reward_ += reward

        episode_duration_ = time.time( ) - self.episode_start_
        if terminal or self.episode_counter_ == 0 :
            self.epoch_reward_ += self.episode_reward_
            self.episode_counter_ += 1

        logging.info( "steps/second %.2f, reward %d, steps %d" % (
                        self.step_counter_ / float( episode_duration_ ),
                        self.episode_reward_, self.step_counter_, ) )
        logging.info( "coaching episode %d: END" % ( self.current_episode_, ) )

    def finish_epoch( self ) :
        assert self.epoch_in_progress_
        self.epoch_in_progress_ = False

        logging.info( "episodes %d, reward %d" % ( self.episode_counter_, self.epoch_reward_, ) )
        logging.info( "coaching epoch %d: END" % ( self.current_epoch_, ) )


if __name__ == "__main__":
    pass
