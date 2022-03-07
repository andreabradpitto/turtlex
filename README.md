# [Turtlex](https://github.com/andreabradpitto/turtlex)

## 📛 Introduction

This is my repository for my Master Thesis project "Safe Learning for Robotics: Abstraction Techniques for Efficient Verification" for the [Robotics Engineering](https://courses.unige.it/10635) master's degree course study (2019-2021), attended at the [University of Genoa](https://unige.it/en).

## 📂 Repository structure

- [doc](doc): folder containing the 2 versions of the assignment specifications

- [src](src): folder containing the source code of the assignment

- [config](config): file storing all the cofiguration settings for the assignment. The header [def.h](src/def.h) contains, among other comments, also the default values for all the elements

- [makefile](makefile): file used to automatically build or delete executables and log files. See [Make](https://en.wikipedia.org/wiki/Make_(software)) for further information

- [.gitignore](.gitignore): hidden file that specifies which files and folder are not relevant for [Git](https://git-scm.com/)

- [LICENSE](LICENSE): a plain text file containing the license terms

- [README.md](README.md): this file

## ❗ Software requirements

- A [**POSIX**](https://en.wikipedia.org/wiki/POSIX)-compliant machine
- [GCC](https://gcc.gnu.org/) compiler collection
- At least 1 MB of free disk space

The space required is intended for the repository contents + the compiled code. Please beware that running the code for an extended time span may produce much greater log file sizes (~ 7 MB per minute with the default [parameters](https://github.com/andreabradpitto/ARP-assignment#configuration-file)).

## ✅ Installation

In order to create the executables, open a terminal, move to this folder, and then run:

```bash
make
```

The make file will take care of compiling all the code needed.  
If you want to remove the executables (and the log file), instead type:

```bash
make clean
```
<!--
## ▶️ Execution

## ℹ️ Additional information

## 📊 Results
-->

## 📫 Author

[Andrea Pitto](https://github.com/andreabradpitto) - s3942710@studenti.unige.it
