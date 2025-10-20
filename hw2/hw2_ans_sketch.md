# 1. Theoretical Peak Performance of CPU
srun -p class1 -N 1 bash -lc 'hostname && lscpu' 명령으로 a01 계산 노드의 정보를 확인하였다.

## (a)
모델명: Intel(R) Xeon(R) Silver 4216 CPU @ 2.10GHz

## (b)
개수: 2

## (c)
base clock frequency: 2.10 GHz
boost clock frequency: 3.20 GHz
전력이나 발열 등의 조건에 따라 안정적으로 동작할 수 있는 성능과 최대 성능을 모두 표현하기 위해 두 종류의 frequency가 있으며, boost clock frequency는 전력이나 발열, 부하 등의 여건이 좋을 때 낼 수 있는 최대 frequency이다.

## (d)
physical 코어 개수: Cores per socket * Socket = 16 * 2 = 32
logical 코어 개수: 64
CPU의 theoretical peak performance를 계산하려면 physical 코어 개수를 사용해야 하는데, 이는 실제로 동시에 연산을 수행할 수 있는 실행 유닛의 개수가 physical 코어의 개수와 같고, 스레드는 물리적인 연산 유닛의 개수를 늘리지 않기 때문이다.

## (e)
Intel Xeon Silver 4216 CPU는 1개의 AVX-512 FMA 유닛을 가지고 있으므로 한 clock cycle 당 실행되는 AVX512 instruction 개수는 1이다.
한 AVX512 instuction으로는 FMA(fused multiply-add) 연산과 512비트 레지스터를 고려하면 32개의 FP32 연산을 수행할 수 있으므로,
하나의 코어는 한 clock cycle에 32개의 FP32 연산을 수행할 수 있다. (TODO: 확인 필요)

## (f)
$R_{peak} = \text{base clock frequncy} \times \text{FP32 Operation/clock cycle} \times \text{number of physical core} = 2.10 \text{GHz} \times 32 \times 32 = 2,150.4 \text{GFLOPS}$


# 2. Matrix Multiplication using PThread
- 병렬화 방식: 행렬 A의 행(row)을 thread의 수에 맞게 분할하여 할당해주는 방식으로 구현하였다. 각 스레드가 행렬 A의 나누어진 부분과, 행렬 B 전체에 접근하여 행렬 C의 나누어진 행을 독립적으로 계산할 수 있도록 하였다. 행렬 B의 열(column)이 아닌, 행렬 A의 열 및 행렬 B의 행애 대해 우선적으로 반복하여 캐시의 spatial locality를 활용할 수 있도록 하였다.

- 스레드 수에 따른 행렬곱 성능: 스레드 수가 1개인 경우와, 16부터 시작하여 16씩 늘려가면서 256개까지 성능을 측정하였다. 시간 문제로 스레드가 1개인 경우는 1회 반복하였고, 나머지 경우는 5회 반복하였다. 이러한 측정을 4번 실행하여 각각의 성능과 평균 성능을 그래프로 나타내었다. 그 결과, 스레드의 개수가 32개가 될 때까지는 성능이 증가하다가 증가와 감소를 반복하고, 이후 점점 하락하는 추세를 그렸다. 이는 스레드 수가 많아질수록 메모리 대역폭이나 스레드 관리 및 스위칭의 오버헤드 등의 성능 하락 요인이 성능 증가 요인보다 더 커지기 때문이다.

- peak performance 대비 실제 성능: 위 그래프에 따르면, 64 스레드에서의 성능이 약 141.49 GFLOPS로 가장 높았다. 이는 1번 (f)에서 구한 2,150.4 GFLOPS의 약 6.6%로, 실제 성능은 이론상 최고 성능과는 큰 차이가 있음을 알 수 있다. 현재 코드에보다 행렬 B와 C에 대한 캐시 활용을 더욱 늘리고, AVX-512와 FMA 연산을 더 적극적으로 활용할 수 있는 라이브러리와 코드를 사용하면 성능을 조금 더 높일 수 있다. 다만, 여전히 이론적인 peak performance보다는 훨씬 낮은 성능을 보이게 될 가능성이 높다.