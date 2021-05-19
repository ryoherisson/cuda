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

# Check clocks
```bash
$ nvidia-smi -a -q -d CLOCK
```

# NVPROF
```bash
# with metrics option
$ nvprof --metrics gld_efficiency,sm_efficiency,achieved_occupacy ./a.out
```

# Dynamic Parallelism
```bash
$ nvcc -arch=sm_75 -rdc=true dynamic_parallelism_check.cu ../common/common.cpp -o ./a.out  -ccbin g++-8
```

# Register Usage
```bash
$ nvcc --ptxas-options=-v register_usage.cu ./a.out -ccbin g++-8
```

# Memory Management
```bash
$ nvprof --print-gpu-trace ./a.out
```

# Enable L1 cache
```bash
$ nvcc -Xptxas -dlcm=ca -o ./a.out misaligned_read.cu
```

# IMage Process with opencv
```bash
$ nvcc sample.cu `pkg-config opencv --cflags --libs` -ccbin g++-8
```