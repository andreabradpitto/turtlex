cmake_minimum_required(VERSION 2.8.3)
project(task_behavior_msgs)

find_package(catkin REQUIRED COMPONENTS message_generation std_msgs)

## Generate messages in the 'msg' folder
add_message_files(
  DIRECTORY msg
  FILES
  NodeData.msg
  TreeDataDump.msg
  TreeNode.msg
  TreeNodeStatus.msg
  TreeStructure.msg
  TreeStatus.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs
)
