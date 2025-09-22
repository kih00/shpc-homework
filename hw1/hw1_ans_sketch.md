# 1. Compilation Process
## 1.1. Preprocessing

### (a)
stdio.h
파일 경로: /usr/include/stdio.h
파일의 라인 수: 875

math.h
파일 경로: /usr/include/math.h
파일의 라인 수: 1341

### (b)
scanf 부분: `extern int scanf (const char *__restrict __format, ...) ;`
printf 부분: `extern int printf (const char *__restrict __format, ...);`
sqrt 부분: `extern double sqrt (double __x) __attribute__ ((__nothrow__ , __leaf__));`

### (c)
Preprocess된 결과에는 실제 구현이 들어있지 않다. 그 이유는 헤더 파일에는 함수에 대한 선언만 들어있기 떄문이다. 실제 구현이 들어있는 라이브러리 파일은 linking 과정에서 object file과 결합된다.

## 1.2. Compilation
### (a)
sqrt.c가 위치한 디렉토리에서: gcc -c sqrt.c

### (b)
sqrt.o가 위치한 디렉토리에서 file sqrt.o를 실행하면 아래와 같은 결과가 나온다.
sqrt/sqrt.o: ELF 64-bit LSB relocatable, x86-64, version 1 (SYSV), not stripped
이를 통해 sqrt.o의 파일 포맷이 ELF (Executable and Linkable Format)라는 것을 알 수 있다.

## 1.3. Linking
### (a)
1.1 (c)에서 언급한 것처럼, sqrt.c를 컴파일한 sqrt.o에는 sqrt 등의 함수에 대한 실제 구현 코드가 없기 때문에 이러한 에러가 발생한다.
따라서, 최종 실행파일을 생성할 때 sqrt가 구현된 별도의 라이브러리(libm.a)를 link하는 옵션인 `-lm`을 명령의 뒷부분에 추가하면 된다.
또한, 최종 실행파일의 이름을 sqrt로 정하기 위해서는 `-o sqrt`를 명령에 추가해야 한다.
따라서 다음과 같은 명령으로 sqrt.o를 올바르게 컴파일하여 sqrt를 생성할 수 있다.
`gcc sqrt.o -o sqrt -lm`

### (b)
스크린샷~~

# 2. C Programming
## 2.1. Shift
### (a)
16(10) = 0000 0000 0000 0000 0000 0000 0001 0000(2)이므로, 32비트 2의 보수표현으로 -16은 이를 반전시킨 1111 1111 1111 1111 1111 1111 1110 1111에 1을 더한 1111 1111 1111 1111 1111 1111 1111 0000이 된다.

### (b)
`a >> 2`의 비트 패턴은, 부호를 반영하여 왼쪽에서 채워지는 부분에 1을 채운 11111111 11111111 11111111 11111100이 된다.

### (c)
`ua >> 2`의 비트 패턴은, 부호를 반영하지 않고 채워지는 부분을 무조건 0으로 차리한 00111111 11111111 11111111 11111100이 된다.

### (d)
Arithmetic Shift는 MSB를 부호로 간주하여, 오른쪽 시프트를 할 때 채워지는 부분을 원래 수의 부호(MSB)와 같도록 한다. 이는 부호를 유지하고, 음수의 시프트도 양수처럼 나눗셈과 같은 효과를 내도록 한다. (a >> b는 a를 2^b로 나눈 몫)
Logical Shift는 채워지는 부분을 반드시 0으로 한다. 이는 연산 대상을 부호 있는 수가 아닌 단순히 비트패턴으로 간주하는 연산으로, 음수의 보수 표현을 가지고 logical shift를 할 경우 나눗셈의 의미를 가지지 않게 된다.

## 2.2. Convert
결과 비교
./convert int 4155
정답: 00000000000000000001000000111011
결과: 00000000000000000001000000111011

./convert long -550
정답: 1111111111111111111111111111111111111111111111111111110111011010
결과: 1111111111111111111111111111111111111111111111111111110111011010

./convert float 3.1415
정답: 01000000010010010000111001010110
결과: 01000000010010010000111001010110

./convert double 3.1415
정답: 0100000000001001001000011100101011000000100000110001001001101111
결과: 0100000000001001001000011100101011000000100000110001001001101111

# 3. 클러스터 사용 연습
## (a)
```
$ sinfo
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
class1       up       1:00      5    mix a[00-04]
class1       up       1:00      7   idle a[05-11]
```

Slurm 노드와 파티션 현황을 확인할 수 있다. 출력에는 class1이라는 파티션의 a[00-11] 노드를 볼 수 있으며, 파티션과 각 노드의 상태 등을 확인할 수 있다.
파티션의 사용 가능 상태(AVAIL)가 up이므로, 파티션에 새로운 작업이 추가될 수 있고 작업이 노드에 할당되어 실행될 수 있다.
작업의 최대 시간 제한(TIMELIMIT)은 1분으로 설정되어 있는 것을 알 수 있다.
각 노드의 상태(STATE)는 5개의 노드가 부분적으로 사용 중(mix)이고, 7개의 노드가 쉬고 있는 중(idle)임을 알 수 있다.

## (b)
```
$ squeue
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
```
Slurm scheduling queue에 배정된 작업 목록을 확인할 수 있다. 위 명령 결과에서는 내용 없이 필드 이름만 출력되었으므로, 현재 할당된 작업이 없음을 의미한다. 만약 'loop'이라는 프로그램을 `srun`으로 계산 노드에 올린 후 `squeue`를 실행하면 아래와 같은 출력을 볼 수 있다.
```
$ squeue
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
           1648472    class1     loop  shpc028 CG       0:37      1 a05
```

## (c)
```
$ srun -p class1 -N 2 hostname
a05
a06
```
srun은 작업을 실행하는 명령어로, -p 옵션으로 파티션을 지정하고 -N 옵션으로 할당할 노드의 수를 정할 수 있다. hostname은 현재 시스템의 호스트 이름을 확인하는 명령이므로, 출력 결과인 a05와 a06은 각각 5번, 6번 노드에서 해당 명령이 실행됨을 의미한다.

## (d)
lscpu 명령은 현재 시스템의 cpu 정보를 보여주는 명령으로, 이를 로그인 노드인 login2에서 직접 실행하면 로그인 노드의 cpu 정보(아키텍쳐, 소켓/코어/스레드 수, 모델명, 캐시 정보, 명령어 세트 등)을 확인할 수 있다.
`srun -p class1 -N 1 lscpu`으로 lscpu를 계산 노드에서 실행하면 계산 노드의 cpu 정보를 확인할 수 있다. 각 노드의 물리적인 하드웨어가 다르기 때문에 두 명령의 결과가 달라진다.