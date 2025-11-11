# HW4

## 1. GPU 정보 확인하기

### (a)
```
$ srun --partition=class1 --gres=gpu:4 nvidia-smi
srun: job 1794159 queued and waiting for resources
srun: job 1794159 has been allocated resources
Mon Nov  3 01:16:30 2025       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 520.61.05    Driver Version: 520.61.05    CUDA Version: 11.8     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA TITAN RTX    On   | 00000000:18:00.0 Off |                  N/A |
| 41%   40C    P8    30W / 280W |      0MiB / 24576MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA TITAN RTX    On   | 00000000:3B:00.0 Off |                  N/A |
| 41%   27C    P8    13W / 280W |      0MiB / 24576MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   2  NVIDIA TITAN RTX    On   | 00000000:86:00.0 Off |                  N/A |
| 41%   25C    P8     8W / 280W |      0MiB / 24576MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   3  NVIDIA TITAN RTX    On   | 00000000:AF:00.0 Off |                  N/A |
| 41%   25C    P8     3W / 280W |      0MiB / 24576MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
계산 노드의 GPU를 4개 할당받아 nvidia-smi 명령(NVIDIA GPU의 현재 상태를 요약하여 보여주는 명령)을 실행

----
```
$ srun --partition=class1 --gres=gpu:4 nvidia-smi -q

==============NVSMI LOG==============

Timestamp                                 : Mon Nov  3 01:24:26 2025
Driver Version                            : 520.61.05
CUDA Version                              : 11.8

Attached GPUs                             : 4
GPU 00000000:18:00.0
    Product Name                          : NVIDIA TITAN RTX
    Product Brand                         : Titan
    Product Architecture                  : Turing
    Display Mode                          : Disabled
    Display Active                        : Disabled
    Persistence Mode                      : Enabled
    MIG Mode
        Current                           : N/A
        Pending                           : N/A
    Accounting Mode                       : Disabled
    Accounting Mode Buffer Size           : 4000
    Driver Model
        Current                           : N/A
        Pending                           : N/A
    Serial Number                         : 1324419051655
    GPU UUID                              : GPU-d7d12c0c-9406-6ae6-beea-8ed0d8d58a71
    Minor Number                          : 0
    VBIOS Version                         : 90.02.2E.00.0C
    MultiGPU Board                        : No
    Board ID                              : 0x1800
    GPU Part Number                       : 900-1G150-2500-000
    Module ID                             : 0
    Inforom Version
        Image Version                     : G001.0000.02.04
        OEM Object                        : 1.1
        ECC Object                        : N/A
        Power Management Object           : N/A
    GPU Operation Mode
        Current                           : N/A
        Pending                           : N/A
    GSP Firmware Version                  : N/A
    GPU Virtualization Mode
        Virtualization Mode               : None
        Host VGPU Mode                    : N/A
    IBMNPU
        Relaxed Ordering Mode             : N/A
    PCI
        Bus                               : 0x18
        Device                            : 0x00
        Domain                            : 0x0000
        Device Id                         : 0x1E0210DE
        Bus Id                            : 00000000:18:00.0
        Sub System Id                     : 0x12A310DE
        GPU Link Info
            PCIe Generation
                Max                       : 3
                Current                   : 1
            Link Width
                Max                       : 16x
                Current                   : 16x
        Bridge Chip
            Type                          : N/A
            Firmware                      : N/A
        Replays Since Reset               : 0
        Replay Number Rollovers           : 0
        Tx Throughput                     : 0 KB/s
        Rx Throughput                     : 0 KB/s
    Fan Speed                             : 41 %
    Performance State                     : P8
    Clocks Throttle Reasons
        Idle                              : Active
        Applications Clocks Setting       : Not Active
        SW Power Cap                      : Not Active
        HW Slowdown                       : Not Active
            HW Thermal Slowdown           : Not Active
            HW Power Brake Slowdown       : Not Active
        Sync Boost                        : Not Active
        SW Thermal Slowdown               : Not Active
        Display Clock Setting             : Not Active
    FB Memory Usage
        Total                             : 24576 MiB
        Reserved                          : 355 MiB
        Used                              : 0 MiB
        Free                              : 24220 MiB
    BAR1 Memory Usage
        Total                             : 256 MiB
        Used                              : 2 MiB
        Free                              : 254 MiB
    Compute Mode                          : Default
    Utilization
        Gpu                               : 0 %
        Memory                            : 0 %
        Encoder                           : 0 %
        Decoder                           : 0 %
    Encoder Stats
        Active Sessions                   : 0
        Average FPS                       : 0
        Average Latency                   : 0
    FBC Stats
        Active Sessions                   : 0
        Average FPS                       : 0
        Average Latency                   : 0
    Ecc Mode
        Current                           : N/A
        Pending                           : N/A
    ECC Errors
        Volatile
            SRAM Correctable              : N/A
            SRAM Uncorrectable            : N/A
            DRAM Correctable              : N/A
            DRAM Uncorrectable            : N/A
        Aggregate
            SRAM Correctable              : N/A
            SRAM Uncorrectable            : N/A
            DRAM Correctable              : N/A
            DRAM Uncorrectable            : N/A
    Retired Pages
        Single Bit ECC                    : N/A
        Double Bit ECC                    : N/A
        Pending Page Blacklist            : N/A
    Remapped Rows                         : N/A
    Temperature
        GPU Current Temp                  : 36 C
        GPU Shutdown Temp                 : 94 C
        GPU Slowdown Temp                 : 91 C
        GPU Max Operating Temp            : 89 C
        GPU Target Temperature            : 84 C
        Memory Current Temp               : N/A
        Memory Max Operating Temp         : N/A
    Power Readings
        Power Management                  : Supported
        Power Draw                        : 30.33 W
        Power Limit                       : 280.00 W
        Default Power Limit               : 280.00 W
        Enforced Power Limit              : 280.00 W
        Min Power Limit                   : 100.00 W
        Max Power Limit                   : 320.00 W
    Clocks
        Graphics                          : 300 MHz
        SM                                : 300 MHz
        Memory                            : 405 MHz
        Video                             : 540 MHz
    Applications Clocks
        Graphics                          : 1350 MHz
        Memory                            : 7001 MHz
    Default Applications Clocks
        Graphics                          : 1350 MHz
        Memory                            : 7001 MHz
    Max Clocks
        Graphics                          : 2100 MHz
        SM                                : 2100 MHz
        Memory                            : 7001 MHz
        Video                             : 1950 MHz
    Max Customer Boost Clocks
        Graphics                          : N/A
    Clock Policy
        Auto Boost                        : N/A
        Auto Boost Default                : N/A
    Voltage
        Graphics                          : N/A
    Processes                             : None

GPU 00000000:3B:00.0
    Product Name                          : NVIDIA TITAN RTX
    Product Brand                         : Titan
...
```
첫 번째 명령과 같지만, -q 옵션을 추가하여 더욱 상세한 정보를 보여줌

```
$ srun --partition=class1 --gres=gpu:4 clinfo
Number of platforms                               1
  Platform Name                                   NVIDIA CUDA
  Platform Vendor                                 NVIDIA Corporation
  Platform Version                                OpenCL 3.0 CUDA 11.8.88
  Platform Profile                                FULL_PROFILE
  Platform Extensions                             cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_fp64 cl_khr_3d_image_writes cl_khr_byte_addressable_store cl_khr_icd cl_khr_gl_sharing cl_nv_compiler_options cl_nv_device_attribute_query cl_nv_pragma_unroll cl_nv_copy_opts cl_nv_create_buffer cl_khr_int64_base_atomics cl_khr_int64_extended_atomics cl_khr_device_uuid cl_khr_pci_bus_info cl_khr_external_semaphore cl_khr_external_memory cl_khr_external_semaphore_opaque_fd cl_khr_external_memory_opaque_fd
  Platform Host timer resolution                  0ns
  Platform Extensions function suffix             NV

  Platform Name                                   NVIDIA CUDA
Number of devices                                 4
  Device Name                                     NVIDIA TITAN RTX
  Device Vendor                                   NVIDIA Corporation
  Device Vendor ID                                0x10de
  Device Version                                  OpenCL 3.0 CUDA
  Driver Version                                  520.61.05
  Device OpenCL C Version                         OpenCL C 1.2 
  Device Type                                     GPU
  Device Topology (NV)                            PCI-E, 18:00.0
  Device Profile                                  FULL_PROFILE
  Device Available                                Yes
  Compiler Available                              Yes
  Linker Available                                Yes
  Max compute units                               72
  Max clock frequency                             1770MHz
  Compute Capability (NV)                         7.5
  Device Partition                                (core)
    Max number of sub-devices                     1
    Supported partition types                     None
    Supported affinity domains                    (n/a)
  Max work item dimensions                        3
  Max work item sizes                             1024x1024x64
  Max work group size                             1024
  Preferred work group size multiple              32
  Warp size (NV)                                  32
  Max sub-groups per work group                   0
  Preferred / native vector sizes                 
    char                                                 1 / 1       
    short                                                1 / 1       
    int                                                  1 / 1       
    long                                                 1 / 1       
    half                                                 0 / 0        (n/a)
    float                                                1 / 1       
    double                                               1 / 1        (cl_khr_fp64)
  Half-precision Floating-point support           (n/a)
  Single-precision Floating-point support         (core)
    Denormals                                     Yes
    Infinity and NANs                             Yes
    Round to nearest                              Yes
    Round to zero                                 Yes
    Round to infinity                             Yes
    IEEE754-2008 fused multiply-add               Yes
    Support is emulated in software               No
    Correctly-rounded divide and sqrt operations  Yes
  Double-precision Floating-point support         (cl_khr_fp64)
    Denormals                                     Yes
    Infinity and NANs                             Yes
    Round to nearest                              Yes
    Round to zero                                 Yes
    Round to infinity                             Yes
    IEEE754-2008 fused multiply-add               Yes
    Support is emulated in software               No
  Address bits                                    64, Little-Endian
  Global memory size                              25396969472 (23.65GiB)
  Error Correction support                        No
  Max memory allocation                           6349242368 (5.913GiB)
  Unified memory for Host and Device              No
  Integrated memory (NV)                          No
  Shared Virtual Memory (SVM) capabilities        (core)
    Coarse-grained buffer sharing                 Yes
    Fine-grained buffer sharing                   No
    Fine-grained system sharing                   No
    Atomics                                       No
  Minimum alignment for any data type             128 bytes
  Alignment of base address                       4096 bits (512 bytes)
  Preferred alignment for atomics                 
    SVM                                           0 bytes
    Global                                        0 bytes
    Local                                         0 bytes
  Max size for global variable                    0
  Preferred total size of global vars             0
  Global Memory cache type                        Read/Write
  Global Memory cache size                        2359296 (2.25MiB)
  Global Memory cache line size                   128 bytes
  Image support                                   Yes
    Max number of samplers per kernel             32
    Max size for 1D images from buffer            268435456 pixels
    Max 1D or 2D image array size                 2048 images
    Max 2D image size                             32768x32768 pixels
    Max 3D image size                             16384x16384x16384 pixels
    Max number of read image args                 256
    Max number of write image args                32
    Max number of read/write image args           0
  Max number of pipe args                         0
  Max active pipe reservations                    0
  Max pipe packet size                            0
  Local memory type                               Local
  Local memory size                               49152 (48KiB)
  Registers per block (NV)                        65536
  Max number of constant args                     9
  Max constant buffer size                        65536 (64KiB)
  Max size of kernel argument                     4352 (4.25KiB)
  Queue properties (on host)                      
    Out-of-order execution                        Yes
    Profiling                                     Yes
  Queue properties (on device)                    
    Out-of-order execution                        No
    Profiling                                     No
    Preferred size                                0
    Max size                                      0
  Max queues on device                            0
  Max events on device                            0
  Prefer user sync for interop                    No
  Profiling timer resolution                      1000ns
  Execution capabilities                          
    Run OpenCL kernels                            Yes
    Run native kernels                            No
    Sub-group independent forward progress        No
    Kernel execution timeout (NV)                 No
  Concurrent copy and kernel execution (NV)       Yes
    Number of async copy engines                  3
    IL version                                    (n/a)
  printf() buffer size                            1048576 (1024KiB)
  Built-in kernels                                (n/a)
  Device Extensions                               cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_fp64 cl_khr_3d_image_writes cl_khr_byte_addressable_store cl_khr_icd cl_khr_gl_sharing cl_nv_compiler_options cl_nv_device_attribute_query cl_nv_pragma_unroll cl_nv_copy_opts cl_nv_create_buffer cl_khr_int64_base_atomics cl_khr_int64_extended_atomics cl_khr_device_uuid cl_khr_pci_bus_info cl_khr_external_semaphore cl_khr_external_memory cl_khr_external_semaphore_opaque_fd cl_khr_external_memory_opaque_fd

  Device Name                                     NVIDIA TITAN RTX
  Device Vendor                                   NVIDIA Corporation
...
```

계산 노드의 GPU를 4개 할당받아 OpenCL 플랫폼 정보와 OpenCL에서 인식한 장치 정보를 보여주는 명령
NVIDIA CUDA라는 플랫폼 이름, 버전, 확장, CPU 정보 등을 확인할 수 있음

### (b)
GPU의 모델명: NVIDIA TITAN RTX
노드당 장착된 GPU의 개수: 4

### (c)
GPU 하나의 메모리 크기: 24576 MiB

### (d)
GPU의 maximum power limit: 280 W
GPU의 maximum SM clock speed: 1770 MHz

### (e)
Max work item dimensions: 3
Max work item sizes: 1024x1024x64 (각 차원의 최대 크기)
Max work group size: 1024

## 2.

### 병렬화 방식
(N * M) 크기의 행렬 C를 정의하기 위해, Kernel index space를 M * N 크기로 정의하여 각 work item (i, j)가 C[i][j]를 계산하도록 하였다.
또한, index space를 TS * TS 크기의 work group들로 나누어 한 work group이 TS * TS 크기의 행렬 C 일부분을 계산하도록 하였다.
이 work group은 계산해야 하는 데이터를 타일 단위(TS)로 쪼개어서 각 그룹이 실행하는 compute unit의 local memory를 최대한 활용할 수 있도록 하였다.
이를 위해 C[i:i+TS][j:j+TS]를 계산하는 데 필요한 총 데이터인 A[i:i+TS][0:K], B[0:K][i:i+TS]의 K차원 부분 또한 타일 단위로 쪼개어 계산한 뒤, 그 계산 결과의 합을 C에 쓰는 것으로 구현하였다.

### 뼈대 코드 설명
\texttt{matmul\_initialize} 함수는 행렬곱을 kernel에서 실행시키기 위한 여러 요소를 생성하고, 행렬곱의 대상이 되는 행렬을 저장할 메모리 버퍼를 생성한다.
clGetPlatformIDs, clGetDeviceIDs을통해 OpenCL 플랫폼과 그 플랫폼에 속한 디바이스 정보를 가져오고, clCreateContext로 실행 환경인 컨택스트를 생성한다.
해당 컨텍스트에서 clCreateCommandQueue, clCreateProgramWithSource로 명령 큐와 프로그램을 생성한 뒤 clBuildProgram로 프룩램