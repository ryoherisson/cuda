# cuda


# Compile on 2070 super
```bash
$ nvcc -arch=sm_75 hello.cu -o ./a.out -ccbin g++-8
```

# Check Warp
```bash
$ nvprof --metrics branch_efficiency ./a.out
# without device optimization
$ nvcc -G -arch=sm_75 hello.cu -o ./a.out -ccbin g++-8
$ nvprof --metrics branch_efficiency ./a.out
```