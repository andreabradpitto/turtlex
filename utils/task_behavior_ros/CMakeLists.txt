cmake_minimum_required(VERSION 2.8.3)
project(task_behavior_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
)

catkin_python_setup()

catkin_package(DEPENDS
  rospy
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# install(PROGRAMS
#   src/task_bahvior_ros/branch.py
#   src/task_bahvior_ros/decorator.py
#   src/task_bahvior_ros/example.py
#   src/task_bahvior_ros/node.py
#   src/task_bahvior_ros/tree.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )
