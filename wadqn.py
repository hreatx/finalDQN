import numpy as np
import tensorflow as tf
import random
import cv2
import gym
import os
from collections import deque
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
LOG_DIR = './log'

class DQN:

    def __init__(self,
                 ur_actions,
                 ur_gamma=0.99,
                 ur_learning_rate=0.00025,
                 ur_epsilon=0.9,
                 ur_epsilon_greedy=True,
                 ur_epsilon_growing_rate=(1.0 - 0.1) / 1000000.0,
                 ur_param_replace_turns=10000,
                 ur_k=4,
                 ur_memory_size=1000000,
                 ur_batch_size=32,
                 ur_begin_train=50000
                 ):
        #self.loss_quick = 0
        self.loss_batch = deque()
        self.k = ur_k
        self.begin_train = ur_begin_train
        self.actions = ur_actions #available action numbers
        ###############hyper parameters
        self.gamma = ur_gamma   #decay factor
        self.learning_rate = ur_learning_rate #learning rate
        ###############epsilon greedy settings
        self.epsilon_max = ur_epsilon
        self.epsilon_greedy = ur_epsilon_greedy
        self.epsilon_growing_rate = ur_epsilon_growing_rate
        self.epsilon = 0 if ur_epsilon_greedy else ur_epsilon #only working epsilon
        ##############cutting off association
        self.param_replace_turns = ur_param_replace_turns   #update target q network turn number
        ############experience replay settings
        self.counter = 0
        self.memory_size = ur_memory_size
        self.batch_size = ur_batch_size
        self.replayMemory = deque()
        self.stateSequence = None   #last four observation preprocessed images to form a state sequence
        #############tensorflow settings
        self.memory_counter = 0
        self.build_net()
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(LOG_DIR)
        self.saver = tf.train.Saver()
        self.sess = tf.Session()  # running session
        self.sess.run(tf.global_variables_initializer())

    def graves_rmsprop_optimizer(self, loss, learning_rate, rmsprop_decay, rmsprop_constant, gradient_clip):
        with tf.name_scope('rmsprop'):
            #optimizer = None
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)

            grads_and_vars = optimizer.compute_gradients(loss)

            grads = []
            params = []
            for p in grads_and_vars:
                if p[0] == None:
                    continue
                grads.append(p[0])
                params.append(p[1])
            # grads = [gv[0] for gv in grads_and_vars]
            # params = [gv[1] for gv in grads_and_vars]
            if gradient_clip > 0:
                grads = tf.clip_by_global_norm(grads, gradient_clip)[0]

            #square_grads = [tf.square(grad) for grad in grads]

            avg_grads = [tf.Variable(tf.zeros(var.get_shape()))
                         for var in params]
            avg_square_grads = [tf.Variable(
                tf.zeros(var.get_shape())) for var in params]

            update_avg_grads = [
                grad_pair[0].assign((rmsprop_decay * grad_pair[0]) + tf.scalar_mul((1 - rmsprop_decay), grad_pair[1]))
                for grad_pair in zip(avg_grads, grads)]
            update_avg_square_grads = [
                grad_pair[0].assign((rmsprop_decay * grad_pair[0]) + ((1 - rmsprop_decay) * tf.square(grad_pair[1])))
                for grad_pair in zip(avg_square_grads, grads)]
            avg_grad_updates = update_avg_grads + update_avg_square_grads

            rms = [tf.sqrt(tf.subtract(avg_grad_pair[1], tf.square(avg_grad_pair[0])) + rmsprop_constant)
                   for avg_grad_pair in zip(avg_grads, avg_square_grads)]

            rms_updates = [grad_rms_pair[0] / grad_rms_pair[1]
                           for grad_rms_pair in zip(grads, rms)]
            train = optimizer.apply_gradients(zip(rms_updates, params))

            return tf.group(train, tf.group(*avg_grad_updates)), grads_and_vars

    def weight_variable(self, shape, c_name, dev):
        initial = tf.truncated_normal(shape=shape, stddev=dev)
        return tf.Variable(initial, collections=c_name)

    def bias_variable(self, shape, c_name):
        initial = tf.constant(1e-4, shape=shape)
        return tf.Variable(initial, collections=c_name)

    def clipped_error(self, x):
        # Huber loss
        try:
            return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
        except:
            return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

    def replace_param(self):
        self.sess.run([tf.assign(self.w1T, self.w1),
                       tf.assign(self.w2T, self.w2),
                       tf.assign(self.w3T, self.w3),
                       tf.assign(self.w4T, self.w4),
                       tf.assign(self.w5T, self.w5),
                       tf.assign(self.b1T, self.b1),
                       tf.assign(self.b2T, self.b2),
                       tf.assign(self.b3T, self.b3),
                       tf.assign(self.b4T, self.b4),
                       tf.assign(self.b5T, self.b5)]
                      )


    def build_net(self):
        self.s = tf.placeholder(tf.float32, shape=[None, 84, 84, 4])
        self.s_next = tf.placeholder(tf.float32, shape=[None, 84, 84, 4])
        self.q_target = tf.placeholder(tf.float32, shape=[None])
        self.a = tf.placeholder(tf.float32, shape=[None, self.actions])
        self.dummy = tf.placeholder(tf.float32)
        self.ep_reward = self.dummy * 1.0
        self.ep_losses = tf.placeholder(tf.float32, shape=[None])
        self.avg_ep_loss = tf.reduce_mean(self.ep_losses)
        tf.summary.scalar('ep_avg_loss', self.avg_ep_loss)
        tf.summary.scalar('episode_reward', self.ep_reward)
        with tf.variable_scope('eval_net'):
            c_name1 = ['eval', tf.GraphKeys.GLOBAL_VARIABLES]
            ###################convolution layer 1##########################
            self.w1 = self.weight_variable([8, 8, 4, 32], c_name1, dev=self.xavier_std(8*8*4, 8*8*32))
            self.b1 = self.bias_variable([32], c_name1)
            l1_conv_relu = tf.nn.relu(tf.nn.conv2d(self.s, self.w1, strides=[1, 4, 4, 1], padding='VALID') + self.b1)
            ###################convolution layer 2##########################
            self.w2 = self.weight_variable([4, 4, 32, 64], c_name1, dev=self.xavier_std(4*4*32, 4*4*64))
            self.b2 = self.bias_variable([64], c_name1)
            l2_conv_relu = tf.nn.relu(tf.nn.conv2d(l1_conv_relu, self.w2, strides=[1, 2, 2, 1], padding='VALID') + self.b2)
            ###################convolution layer 2##########################
            self.w3 = self.weight_variable([3, 3, 64, 64], c_name1, dev=self.xavier_std(3*3*64, 3*3*64))
            self.b3 = self.bias_variable([64], c_name1)
            l3_conv_relu = tf.nn.relu(tf.nn.conv2d(l2_conv_relu, self.w3, strides=[1, 1, 1, 1], padding='VALID') + self.b3)
            ###################fully connected layer 1######################
            l3_reshape = tf.reshape(l3_conv_relu, [-1, 3136])
            self.w4 = self.weight_variable([3136, 512], c_name1, dev=self.xavier_std(3136, 512))
            self.b4 = self.bias_variable([512], c_name1)
            l4_full_connect = tf.nn.relu(tf.matmul(l3_reshape, self.w4) + self.b4)
            ###################output fully connected layer 1###############
            self.w5 = self.weight_variable([512, self.actions], c_name1, dev=self.xavier_std(512, self.actions))
            self.b5 = self.bias_variable([self.actions], c_name1)
            self.q_eval = tf.matmul(l4_full_connect, self.w5) + self.b5
        ###################loss tensor##################################
        self.q_mediate = tf.reduce_sum(tf.multiply(self.q_eval, self.a), 1)
        self.diff = tf.subtract(self.q_mediate, self.q_target)
        self.squared_loss = tf.reduce_mean(tf.square(self.diff))
        self.loss = tf.reduce_mean(self.clipped_error(self.diff))

        #self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_mediate))
        self.learn_tensor, grads = self.graves_rmsprop_optimizer(self.loss, self.learning_rate, 0.9, 0.01, 1)
        with tf.variable_scope('target_net'):
            c_name2 = ['target', tf.GraphKeys.GLOBAL_VARIABLES]
            ###################convolution layer 1##########################
            self.w1T = self.weight_variable([8, 8, 4, 32], c_name2, dev=self.xavier_std(8*8*4, 8*8*32))
            self.b1T = self.bias_variable([32], c_name2)
            l1_conv_reluT = tf.nn.relu(tf.nn.conv2d(self.s_next, self.w1T, strides=[1, 4, 4, 1], padding='VALID') + self.b1T)
            ###################convolution layer 2##########################
            self.w2T = self.weight_variable([4, 4, 32, 64], c_name2, dev=self.xavier_std(4*4*32, 4*4*64))
            self.b2T = self.bias_variable([64], c_name2)
            l2_conv_reluT = tf.nn.relu(tf.nn.conv2d(l1_conv_reluT, self.w2T, strides=[1, 2, 2, 1], padding='VALID') + self.b2T)
            ###################convolution layer 2##########################
            self.w3T = self.weight_variable([3, 3, 64, 64], c_name2, dev=self.xavier_std(3*3*64, 3*3*64))
            self.b3T = self.bias_variable([64], c_name2)
            l3_conv_reluT = tf.nn.relu(tf.nn.conv2d(l2_conv_reluT, self.w3T, strides=[1, 1, 1, 1], padding='VALID') + self.b3T)
            ###################fully connected layer 1######################
            l3_reshapeT = tf.reshape(l3_conv_reluT, [-1, 3136])
            self.w4T = self.weight_variable([3136, 512], c_name2, dev=self.xavier_std(3136, 512))
            self.b4T = self.bias_variable([512], c_name2)
            l4_full_connectT = tf.nn.relu(tf.matmul(l3_reshapeT, self.w4T) + self.b4T)
            ###################output fully connected layer 1###############
            self.w5T = self.weight_variable([512, self.actions], c_name2, dev=self.xavier_std(512, self.actions))
            self.b5T = self.bias_variable([self.actions], c_name2)
            self.q_next = tf.matmul(l4_full_connectT, self.w5T) + self.b5T

    def clipped_reward(self, reward):
        return max(-1, min(reward, 1))




    # def convert_RGB_to_Gray(self, RGB):
    #     return (299.0 * RGB[0] + 587.0 * RGB[1] + 114.0 * RGB[2]) / 1000.0
    #
    # def preprocess(self, image_raw):
    #     high, broad, big_three = image_raw.shape
    #     # print(high, broad, big_three)
    #     image_inter = np.zeros((high, broad))
    #
    #     for j in range(high):
    #         for k in range(broad):
    #             image_inter[j, k] = self.convert_RGB_to_Gray(image_raw[j, k])
    #             # if j == 1 and k == 1:
    #             #     prin
    #     print(image_inter.shape)
    #     image_preprocessed = block_reduce(image_inter, block_size=(110,84), func=np.max)
    #     print(image_preprocessed.shape)
    #     image_preprocessed = image_preprocessed[27:110, :]
    #     print(image_preprocessed.shape)
    #     return np.asarray(image_preprocessed)
    def xavier_std(self, in_size, out_size):
        return np.sqrt(2. / (in_size + out_size))

    def preprocess(self, image_raw):
        #previous_frame = self.stateSequence[:, :, 3]
        image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY)
        image_raw = cv2.resize(image_raw, (84,84))
        ret, image_raw = cv2.threshold(image_raw, 1, 255, cv2.THRESH_BINARY)
        return image_raw

    def store_transition(self, newob, action, reward, done):
        # if not hasattr(self, 'memory_counter'):
        #     self.memory_counter = 0
        #print(newob.shape)

        # print(self.stateSequence.shape)
        newstate = np.stack(newob, axis=2)
        #print(newstate.shape)
        transition = [self.stateSequence, action, reward, newstate, done]
        self.replayMemory.append(transition)
        if len(self.replayMemory) > self.memory_size:
            self.replayMemory.popleft()
        if self.memory_counter < self.begin_train:
            self.memory_counter += 1
        # if self.memory_counter % 100 == 0:
        #     print self.memory_counter
        self.stateSequence = newstate

    def set_init_state(self, ob):
        #print('set_init_state: ', ob.shape)
        self.stateSequence = np.stack(ob, axis=2)
        #print('set_init_state: ', self.stateSequence.shape)
        # print('initial state shape:', self.stateSequence.shape)

    def choose_action(self):
        ob = self.stateSequence
        if np.random.uniform() < self.epsilon:
            q_approx = self.sess.run(self.q_eval, feed_dict={self.s: ob[np.newaxis, :]})
            act = np.argmax(q_approx)
        else:
            act = random.randint(0, self.actions - 1)


        return act

    def get_summary(self, ep_reward, ep_losses):
        summ = self.sess.run(self.merged, feed_dict={
            self.dummy: ep_reward,
            self.ep_losses: ep_losses

        })
        return summ

    def learn(self):
        self.counter += 1
        if self.counter == 1:
            print('begin training.')
        if self.counter % self.param_replace_turns == 1:
            self.replace_param()
        # sample_index = np.random.choice(self.memory_size, self.batch_size)
        batch = random.sample(self.replayMemory, self.batch_size)
        batch_pre_state = [data[0] for data in batch]
        batch_action = [data[1] for data in batch]
        batch_reward = [data[2] for data in batch]
        batch_post_state = [data[3] for data in batch]
        batch_done = [data[4] for data in batch]
        # Q_eval = self.sess.run(self.q_eval, feed_dict={self.s: batch_pre_state})
        Q_next = self.sess.run(self.q_next, feed_dict={self.s_next: batch_post_state})
        Q_target = np.zeros(self.batch_size)
        action_array = np.zeros((self.batch_size, self.actions))
        for i in range(self.batch_size):
            if not batch_done[i]:
                Q_target[i] = batch_reward[i] + self.gamma * np.max(Q_next[i])
            else:
                Q_target[i] = batch_reward[i]
            action_array[i][batch_action[i]] = 1
        #self.loss_quick = self.sess.run(self.loss, feed_dict={})
        a_loss, _ = self.sess.run([self.squared_loss, self.learn_tensor], feed_dict={self.s: batch_pre_state, self.q_target: Q_target, self.a: action_array})
        self.loss_batch.append(a_loss)
        if self.epsilon_greedy:
            if self.epsilon < self.epsilon_max:
                self.epsilon += self.epsilon_growing_rate
        if self.counter > 100 and self.counter % 1000000 == 0:
            self.saver.save(self.sess, 'modelIn{}'.format(self.counter), global_step=self.counter)
            print('model saved for loop {}.'.format(self.counter))
            print('\n')



TRAIN_EPISODES = 20000

def main():
    gym.envs.register(id='bo-v0', entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': 'breakout', 'obs_type': 'image', 'frameskip': 4, 'repeat_action_probability': 0.0},
            max_episode_steps=100000,
            nondeterministic=False,)
    env = gym.make('bo-v0')
    env = env.unwrapped
    DRL = DQN(4)
    total_steps = 0
    begin_learn = False
    for episode in range(TRAIN_EPISODES):
        reset_counter = 5
        # done_flag = 0
        total = 0
        steps = 0

        #total_action = deque()
        ob = env.reset()
        # for i in xrange(random.randint(4, 29)):
        #     lives = env.ale.lives()
        #     ob, _, done, _ = env.step(0)
        #     lost_one_live_1 = (not lives == env.ale.lives())
        #     if lost_one_live_1 or done:
        #         env.reset()
        #     preprocessed_ob = DRL.preprocess(ob)


        preprocessed_ob = DRL.preprocess(ob)
        ob_sequence = deque()
        ob_sequence.append(preprocessed_ob)
        ob_sequence.append(preprocessed_ob)
        ob_sequence.append(preprocessed_ob)
        ob_sequence.append(preprocessed_ob)
        DRL.set_init_state(ob_sequence)
        #action = env.action_space.sample()
        while True:
            store_flage = True
            #env.render()
            #pre_action = action
            action = DRL.choose_action()
            lives = env.ale.lives()
            newob, reward, done, info = env.step(action)
            lost_one_live = (not lives == env.ale.lives())
            total += reward
            preprocessed_newob = DRL.preprocess(newob)
            ob_sequence.append(preprocessed_newob)
            ob_sequence.popleft()

            if steps < 4 or reset_counter < 4:
                store_flage = False
                DRL.set_init_state(ob_sequence)
                reset_counter += 1
            # if len(ob_sequence) > 4:
            #     ob_sequence.popleft()
            if lost_one_live:
                reset_counter = 0
            steps += 1
            if store_flage:
                DRL.store_transition(ob_sequence, action, DRL.clipped_reward(reward), done or lost_one_live)
            if DRL.memory_counter >= DRL.begin_train:
                DRL.learn()
                total_steps += 1
                begin_learn = True
            if done:
                # done_flag += 1
                print('episode {} reward: {}, episode steps: {}, total steps: {}, epilson: {}'.format(episode, total, steps, DRL.counter,
                                                                                                      DRL.epsilon))
                if begin_learn:
                    summ = DRL.get_summary(total, DRL.loss_batch)
                    DRL.loss_batch = deque()
                    DRL.writer.add_summary(summ, episode)
                break





if  __name__ == '__main__':
    main()










