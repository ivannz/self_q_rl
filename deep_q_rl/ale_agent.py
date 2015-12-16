"""
The NeuralAgent class wraps a deep Q-network for training and testing
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
sys.setrecursionlimit(10000)

class NeuralAgent(object):

    def __init__(self, q_network, epsilon_start, epsilon_min,
                 epsilon_decay, replay_memory_size, exp_pref,
                 replay_start_size, update_frequency, rng):

        self.network = q_network
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.replay_memory_size = replay_memory_size
        self.exp_pref = exp_pref
        self.replay_start_size = replay_start_size
        self.update_frequency = update_frequency
        self.random_state = rng

## Remember the dimensionality of the input space
        self.phi_length = self.network.num_frames
        self.image_width = self.network.input_width
        self.image_height = self.network.input_height
## The output lyaer size of the q-value network approximator
        self.num_actions = self.network.num_actions

## Allocate experience replay datasets: a large one for trainig ...
        self.dataset_training = ale_data_set.DataSet( width = self.image_width, height = self.image_height,
                                                      rng = self.random_state,
                                                      max_steps = self.replay_memory_size,
                                                      phi_length = self.phi_length )
##   ... and a small one for testing.
## Since during the testing pahse no learning takes place, we just need
##  this dataset to be big enough to hold the current phi ( state x phi_length ).
##  Thus "max_steps" is set to double the size of phi.
        self.dataset_testing = ale_data_set.DataSet( width = self.image_width, height = self.image_height,
                                                     rng = self.random_state,
                                                     max_steps = self.phi_length * 2,
                                                     phi_length = self.phi_length )

## The epsilon-probability changes across the epochs!
        self.epsilon = self.epsilon_start
        if self.epsilon_decay != 0 :
            self.epsilon_rate = ( ( self.epsilon_start - self.epsilon_min ) / self.epsilon_decay )
        else :
            self.epsilon_rate = 0

        # CREATE A FOLDER TO HOLD THE RESULTS
        time_str = time.strftime("_%m-%d-%H-%M", time.gmtime())
        self.exp_dir = self.exp_pref + time_str +  "_" \
                       + "{}".format( self.network.lr ).replace( ".", "p" ) + "_" \
                       + "{}".format( self.network.discount ).replace( ".", "p" )

        try:
            os.stat(self.exp_dir)
        except OSError:
            os.makedirs(self.exp_dir)

        self._open_results_file( )
        self._open_learning_file( )

        self.holdout_observations = None
## Logging:
        logging.info( "NeuralAgent: actions %d, phi %d" % ( self.num_actions, self.phi_length, ) )

## Handy (?) functions to open and create the files
    def _open_results_file( self ) :
        file_ = os.path.join( self.exp_dir, 'results.csv' )
        logging.info( "OPENING " + file_ )
        self.f_results_ = open( file_, 'w', 0 )
        self.f_results_.write( 'epoch,num_episodes,total_reward,reward_per_episode,mean_q\n' )
        self.f_results_.flush( )

    def _open_learning_file( self ) :
        file_ = os.path.join( self.exp_dir, 'learning.csv' )
        logging.info( "OPENING " + file_ )
        self.f_learning_ = open( file_, 'w', 0 )
        self.f_learning_.write( 'mean_loss,epsilon\n' )
        self.f_learning_.flush( )

## Show the states ( phi )
    def _show_phis(self, phi1, phi2):
        import matplotlib.pyplot as plt
        for p in range(self.phi_length):
            plt.subplot(2, self.phi_length, p+1)
            plt.imshow(phi1[p, :, :], interpolation='none', cmap="gray")
            plt.grid(color='r', linestyle='-', linewidth=1)
        for p in range(self.phi_length):
            plt.subplot(2, self.phi_length, p+5)
            plt.imshow(phi2[p, :, :], interpolation='none', cmap="gray")
            plt.grid(color='r', linestyle='-', linewidth=1)
        plt.show()

## Pickle the trained network
    def save( self, name ) :
        with open( os.path.join( self.exp_dir, name ), 'w' ) as fout :
                cPickle.dump( self.network, fout, -1 )

## Epochs: start/finish procedures
    def start_epoch( self, epoch, testing ) :
        assert not ( hasattr( self, 'epoch_in_progress_' ) and self.epoch_in_progress_ )
        self.epoch_in_progress_ = True
## Logging:
        logging.info( "epoch %d: START (testing = %c)" % ( epoch, "T" if testing else "F" ) )
## Keep information on the current epoch
        self.current_epoch_, self.testing_ = epoch, testing
        self.epoch_start_ = time.time( )
## Prepare the internal state for the epoch
        self.epoch_reward_, self.episode_counter_ = 0, 0
## Preselect the dataset
        self.current_dataset_ = self.dataset_testing if self.testing_ else self.dataset_training

    def finish_epoch( self ) :
        assert self.epoch_in_progress_
        self.epoch_in_progress_ = False
## End the current epoch
        if not self.testing_ :
            self.save( "network_file_%d.pkl" % ( self.current_epoch_, ) )
        else :
## (?) Why is this fixed? And why is training experience used for estimating Q-values?
            holdout_size = 3200
            if self.holdout_observations is None and len( self.dataset_training ) > holdout_size :
## The hold out state set is initialized once and never altered again (at least automatically).
                self.holdout_observations = self.dataset_training.random_batch( holdout_size )[ 0 ]
## Compute the optimal Q-value
            logging.info( "epoch %d: computing average Q-val" % ( self.current_epoch_, ) )
            holdout_sum = 0
            if self.holdout_observations is not None :
                for i in range( holdout_size ) :
                    holdout_sum += np.max( self.network.q_vals( self.holdout_observations[ i, ... ] ) )
## Write the results of testing
            self.f_results_.write( "{},{},{},{},{}\n".format(
                                        self.current_epoch_, self.episode_counter_,
                                        self.epoch_reward_, self.epoch_reward_ / float( self.episode_counter_ ),
                                        holdout_sum / holdout_size ) )
            self.f_results_.flush( )
## Logging:
        logging.info( "episodes %d, reward %d" % ( self.episode_counter_, self.epoch_reward_, ) )
        logging.info( "epoch %d: END" % ( self.current_epoch_, ) )

## Episodes: start/finish procedures
    def start_episode( self, observation ):
        """
        This method is called once at the beginning of each episode.
        No reward is provided, because reward is only available after
        an action has been taken.

        Arguments:
           observation - height x width numpy array

        Returns:
           An integer action
        """
        assert self.epoch_in_progress_
        assert not ( hasattr( self, 'episode_in_progress_' ) and self.episode_in_progress_ )
        self.episode_in_progress_ = True
## Logging:
        logging.info( "episode %d: START %f" % ( self.episode_counter_, self.epsilon ) )
## Reset per episode counters and the mean loss for every episode.
        self.step_counter_, self.episode_reward_ = 0, 0
        self.current_episode_ = self.episode_counter_
        self.episode_start_ = time.time( )
        self.episode_training_loss_ = [ ]
## Initialize with an action chosen uniformly at random from the action space (epsilon = 1.0).
        action = self._choose_action( self.current_dataset_, 1.0, observation )
## Store the first state-action pair to act upon next.
# In order to add an element to the data set we need the previous state and
#  action and the current reward. These will be used to store states and actions.
        self.last_action_, self.last_observation_ = action, observation
## IDEA: maybe it is better to fill at most #phi states with "blanks"
##  in order to avoid the episode border effects.
## IDEA: what if the network has an extra input that tells the net
##  if the policy was terminal.
## IDEA: maybe add dropout? But where?
        return action

    def end_episode(self, reward, terminal = True ):
        """
        This function is called once at the end of an episode.

        Arguments:
           reward      - Real valued reward.
           terminal    - Whether the episode ended intrinsically
                         (ie we didn't run out of steps)
        Returns:
            None
        """
        assert self.epoch_in_progress_ and self.episode_in_progress_
        self.episode_in_progress_ = False
## Record the last experience (state-action-reward), but indicate that it was terminal.
        self.current_dataset_.add_sample( self.last_observation_, self.last_action_,
                                          np.clip( reward, -1, 1 ), terminal = True )
## End the episode
        self.step_counter_ += 1
        self.episode_reward_ += reward
        episode_duration_ = time.time( ) - self.episode_start_
## Count only full episodes unless the current is the only episode
        if terminal or self.episode_counter_ == 0 :
            self.epoch_reward_ += self.episode_reward_
            self.episode_counter_ += 1
## Log the speed
        logging.info( "steps/second %.2f, reward %d, steps %d" % (
                        self.step_counter_ / float( episode_duration_ ),
                        self.episode_reward_, self.step_counter_, ) )
        # logging.info( "experience %d / %d / %d" % ( len( self.current_dataset_ ),
        #                                             self.replay_start_size,
        #                                             self.replay_memory_size, ) )
        # logging.info( "steps %d, reward %d" % ( self.step_counter_, self.episode_reward_, ) )
## And the average training loss with the last epsilon
        if not self.testing_ and ( len( self.episode_training_loss_ ) > 0 ) :
            average_loss_ = np.mean( self.episode_training_loss_ )
            logging.info( "average loss: {:.4f}".format( average_loss_ ) )
            self.f_learning_.write( "{},{}\n".format( average_loss_, self.epsilon ) )
            self.f_learning_.flush( )
## Logging:
        logging.info( "episode %d: END" % ( self.current_episode_, ) )

## Perform one step in an episode: record the reward and the action
    def step( self, reward, observation ) :
        """
        This method is called each time step.

        Arguments:
           reward      - Real valued reward.
           observation - A height x width numpy array

        Returns:
           An integer action.

        """
        assert self.episode_in_progress_
## Store the most recent data: state, action and the clipped reward.
        has_enough_experience = ( len( self.current_dataset_ ) > self.replay_start_size )
## Get the expected present values of the future reward stream for the $\epsilon$-greedy
##  policy defined by:
##       \hat{a} = \argmax_{k\in \Acal} Q( s_t, k | \theta_t ) \,,
##  wheree $\theta_t$ is piece-wise consant adapted parameter process.
        ## prio = self._expected_reward( self.current_dataset_, self.last_observation_,
        ##                               self.last_action_, np.clip( reward, -1, 1 ) )
        self.current_dataset_.add_sample( self.last_observation_, self.last_action_,
                                          np.clip( reward, -1, 1 ), terminal = False )
## Training and testing are actually the same, except for the former also
##   performing a deep Q-learning step.
        self.episode_reward_ += reward
        self.step_counter_ += 1
## During testing phase we accumulate the reward and set the $\epsilon$ to 0.05
        if self.testing_ :
            epsilon_ = 0.05
        else :
## If there is enough data to learn from, update the probability of
##  a random action. This way the agent initally explores the state-action
##  space, and only at later steps does it act more "rationally". Right
##  now the probability decreases steadily in a linear fashion.
            if has_enough_experience :
## IDEA: maybe use a "cooling" schedule like in simulated annealing, or
##       make random action probability depend on the reward.
                self.epsilon = max( self.epsilon_min, self.epsilon - self.epsilon_rate )
## During the trainig phase, there are two modes: training experience
##  accumulation and training.
            epsilon_ = self.epsilon

## Exectue an action: follow an $\epsilon$-greedy strategy.
## IDEA: add an extra network which would decide the $\epsilon$ -- like
##       a skill confidence network.
        action = self._choose_action( self.current_dataset_, epsilon_, observation )

## Now if we are currently training and there is enough experience to replay ...
        if not self.testing_ and has_enough_experience :
##  ... check if it is high time to actually learn something.
            if self.step_counter_ % self.update_frequency == 0 :
                loss_ = self._do_training( self.current_dataset_ )
                self.episode_training_loss_.append( loss_ )
## Return
        self.last_action_, self.last_observation_ = action, observation
        return action

## Implements an $\epsilon$-greedy strategy.
    def _choose_action( self, dataset, epsilon, observation ) :
## Fallback to a random strategy if there is not enough data to base the decision upon,
##   of if it is time to apply and $\epsilon$-greedy part.
        if ( self.random_state.rand( ) < epsilon ) or ( self.step_counter_ < self.phi_length ) :
## Pick an action _uniformly_ at random from the action space.
            return self.random_state.randint( 0, self.num_actions )
## Choose an action based on the current policy, but in an $\epsilon$-greedy manner.
        return self.network.choose_action( dataset.phi( observation ) )

    def _do_training( self, dataset ) :
        """
        Returns the average loss for the current batch.
        May be overridden if a subclass needs to train the network
        differently.
        """
        states, actions, rewards, next_states, terminals = \
                                dataset.random_batch( self.network.batch_size )
        return self.network.train( states, actions, rewards, next_states, terminals )


if __name__ == "__main__":
    pass
