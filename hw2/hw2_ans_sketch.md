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