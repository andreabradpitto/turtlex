# [Turtlex](https://github.com/andreabradpitto/turtlex)

## 📛 Introduction

This is my repository for my Master Thesis project "Safe Learning for Robotics: Abstraction Techniques for Efficient Verification" for the [Robotics Engineering](https://courses.unige.it/10635) master's degree course study (2019-2021), attended at the [University of Genoa](https://unige.it/en).

## 📂 Repository structure

- [task_behavior_engine](task_behavior_engine), [task_behavior_msgs](task_behavior_msgs), [task_behavior_ros](task_behavior_ros): forked from [Toyota Research Institute](https://github.com/ToyotaResearchInstitute)

- [rosnode](rosnode): forked from [ros_comm](https://github.com/ros/ros_comm)

- [turtlex](turtlex): TODO

- [.gitignore](.gitignore): hidden file that specifies which files and folders are not relevant for [Git](https://git-scm.com/)

- [LICENSE](LICENSE): a plain text file containing the license terms

- [README.md](README.md): this file

## ❗ Software requirements

- [ROS Noetic](http://wiki.ros.org/noetic/Installation) (full installation recommended)
- [openai_ros](https://bitbucket.org/theconstructcore/openai_ros/src/kinetic-devel/)
- [OpenaAI Gym](https://gym.openai.com/docs/)
- [PyTorch](https://pytorch.org/get-started/locally/)
- [TensorFlow](https://www.tensorflow.org/install/)
- [OpenCV](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html)
- [MPI fo Python](https://mpi4py.readthedocs.io/en/stable/install.html) (`python -m pip install mpi4py`)
- [pyassimp](https://pypi.org/project/pyassimp/) (`python -m pip install pyassimp`) [due to this issue](https://github.com/ros-planning/moveit/issues/86)

## ✅ Installation

In order to create the executables, open a terminal, move to this folder, and then run:

```bash
cd catkin_ws/src
git clone https://github.com/andreabradpitto/turtlex.git
cd ..
catkin_make
```

## ▶️ Execution

After re-sourcing (i.e., move back to the `catkin_ws` folder with `cd ..`, then type `source devel/setup.bash` and confirm) or opening a new terminal emulator, execute:

```bash
roslaunch turtlex_bt bt.launch
```

## 📫 Author

[Andrea Pitto](https://github.com/andreabradpitto) - s3942710@studenti.unige.it
