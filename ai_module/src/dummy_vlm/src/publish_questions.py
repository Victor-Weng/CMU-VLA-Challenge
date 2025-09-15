#!/usr/bin/env python3
import rospy
from std_msgs.msg import String

rospy.init_node('question_publisher')
pub = rospy.Publisher('/challenge_question', String, queue_size=10)

while not rospy.is_shutdown():
    question = input("Type your question: ")
    pub.publish(question)