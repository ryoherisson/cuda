# Compile on 2070 super
```bash
$ nvcc -arch=sm_75 hello.cu -o hello -ccbin g++-8
```