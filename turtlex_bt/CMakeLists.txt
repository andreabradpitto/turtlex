cmake_minimum_required(VERSION 2.8.3)
project(turtlex_bt)

find_package(catkin REQUIRED COMPONENTS rospy task_behavior_ros)

catkin_package(
   CATKIN_DEPENDS rospy task_behavior_ros 
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS
  scripts/*.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
