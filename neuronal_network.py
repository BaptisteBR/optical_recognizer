# Author: Baptiste BRIOT--RIBEYRE
# <baptiste.briot--ribeyre@alumni.univ-avignon.fr>

"""
"""



from __future__ import print_function
import tensorflow as tf

if __name__ == "__main__":
    
    # Create nodes 1 & 2
    node1 = tf.constant(3.0, dtype=tf.float32)
    node2 = tf.constant(4.0)
    
    print(node1, node2)
    
    # Get session
    sess = tf.Session()
    
    print(sess.run([node1, node2]))
    
    # Create node 3
    node3 = tf.add(node1, node2)
    
    print("node3:", node3)
    
    print("sess.run(node3):", sess.run(node3))