cmake_minimum_required(VERSION 2.8.3)
project(task_behavior_engine)

find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# install(PROGRAMS
#   src/task_bahvior_engine/branch.py
#   src/task_bahvior_engine/decorator.py
#   src/task_bahvior_engine/example.py
#   src/task_bahvior_engine/node.py
#   src/task_bahvior_engine/tree.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )
