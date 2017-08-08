import tensorflow as tf
import numpy as np
import gym
import cv2
import random
from collections import deque

TEST = 1000


def preprocess(image_raw):
    # previous_frame = self.stateSequence[:, :, 3]
    image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY)
    image_raw = cv2.resize(image_raw, (84, 84))
    ret, image_raw = cv2.threshold(image_raw, 1, 255, cv2.THRESH_BINARY)
    return image_raw

def main():
    sess = tf.Session()
    model = tf.train.import_meta_graph('modelIn7000000-7000000.meta')
    model.restore(sess, 'modelIn7000000-7000000')
    eval_net = tf.get_collection('eval')
    w1 = eval_net[0]
    b1 = eval_net[1]
    w2 = eval_net[2]
    b2 = eval_net[3]
    w3 = eval_net[4]
    b3 = eval_net[5]
    w4 = eval_net[6]
    b4 = eval_net[7]
    w5 = eval_net[8]
    b5 = eval_net[9]
    s = tf.placeholder(tf.float32, shape=[None, 84, 84, 4])
    #s_next = tf.placeholder(tf.float32, shape=[None, 84, 84, 4])
    #q_target = tf.placeholder(tf.float32, shape=[None])
    #a = tf.placeholder(tf.float32, shape=[None, 4])
    with tf.variable_scope('eval_net'):
        #c_name1 = ['eval', tf.GraphKeys.GLOBAL_VARIABLES]
        ###################convolution layer 1##########################
        #w1 = weight_variable([8, 8, 4, 32], c_name1, dev=xavier_std(8 * 8 * 4, 8 * 8 * 32))
        #b1 = bias_variable([32], c_name1)
        l1_conv_relu = tf.nn.relu(tf.nn.conv2d(s, w1, strides=[1, 4, 4, 1], padding='VALID') + b1)
        ###################convolution layer 2##########################
        #w2 = weight_variable([4, 4, 32, 64], c_name1, dev=xavier_std(4 * 4 * 32, 4 * 4 * 64))
        #b2 = bias_variable([64], c_name1)
        l2_conv_relu = tf.nn.relu(tf.nn.conv2d(l1_conv_relu, w2, strides=[1, 2, 2, 1], padding='VALID') + b2)
        ###################convolution layer 2##########################
        #w3 = weight_variable([3, 3, 64, 64], c_name1, dev=xavier_std(3 * 3 * 64, 3 * 3 * 64))
        #b3 = bias_variable([64], c_name1)
        l3_conv_relu = tf.nn.relu(tf.nn.conv2d(l2_conv_relu, w3, strides=[1, 1, 1, 1], padding='VALID') + b3)
        ###################fully connected layer 1######################
        l3_reshape = tf.reshape(l3_conv_relu, [-1, 3136])
        #w4 = weight_variable([3136, 512], c_name1, dev=xavier_std(3136, 512))
        #b4 = bias_variable([512], c_name1)
        l4_full_connect = tf.nn.relu(tf.matmul(l3_reshape, w4) + b4)
        ###################output fully connected layer 1###############
        #w5 = weight_variable([512, actions], c_name1, dev=xavier_std(512, actions))
        #b5 = bias_variable([actions], c_name1)
        q_eval = tf.matmul(l4_full_connect, w5) + b5

    gym.envs.register(id='bo-v0', entry_point='gym.envs.atari:AtariEnv',
                      kwargs={'game': 'breakout', 'obs_type': 'image', 'frameskip': 4,
                              'repeat_action_probability': 0.0},
                      max_episode_steps=100000,
                      nondeterministic=False, )
    env = gym.make('bo-v0')
    env = env.unwrapped
    #DRL = DQN(4)
    total_steps = 0

    def choose_action(ob):
        if np.random.uniform() < 0.95:
            act = np.argmax(sess.run(q_eval, feed_dict={s: ob[np.newaxis, :]}))
        else:
            act = env.action_space.sample()
        return act

    for episode in range(TEST):
        total = 0
        steps = 0
        ob = env.reset()
        preprocessed_ob = preprocess(ob)
        ob_sequence = deque()
        ob_sequence.append(preprocessed_ob)
        ob_sequence.append(preprocessed_ob)
        ob_sequence.append(preprocessed_ob)
        ob_sequence.append(preprocessed_ob)
        feed_ob_sequence = np.stack(ob_sequence, axis=2)
        while True:
            env.render()
            #q_value = sess.run(q_eval, feed_dict={s: feed_ob_sequence[np.newaxis, :]})
            action = choose_action(feed_ob_sequence)
            newob, reward, done, info = env.step(action)
            total += reward
            #print(done)
            preprocessed_newob = preprocess(newob)
            ob_sequence.append(preprocessed_newob)
            ob_sequence.popleft()
            feed_ob_sequence = np.stack(ob_sequence, axis=2)
            steps += 1
            total_steps += 1
            if done:
                #print(done)
                print('episode {} reward: {}, episode steps: {}'.format(episode, total, steps))
                break


if __name__ == '__main__':
    main()