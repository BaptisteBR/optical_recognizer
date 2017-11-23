#! /usr/bin/python
# -*- coding: utf-8 -*-

# Author: Baptiste BRIOT--RIBEYRE
# <baptiste.briot--ribeyre@alumni.univ-avignon.fr>

"""
"""

import tensorflow as tf

if __name__ == "__main__":
    
    # Create nodes
    node1 = tf.constant(3.0, dtype=tf.float32)
    node2 = tf.constant(4.0)
    
    print(node1, node2)
    
    # Get session
    sess = tf.Session()
    
    print(sess.run([node1, node2]))