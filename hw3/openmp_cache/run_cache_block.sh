make

num_threads=$1
mat_size=$2

for i in {16..240..16}; do
  ./run.sh -v -t $num_threads -n 3 $mat_size $mat_size $mat_size $i
  echo "================================"
  sleep 2
done

for i in {256..960..64}; do
  ./run.sh -v -t $num_threads -n 3 $mat_size $mat_size $mat_size $i
  echo "================================"
  sleep 2
done

for i in {1024..2304..128}; do
  ./run.sh -v -t $num_threads -n 3 $mat_size $mat_size $mat_size $i
  echo "================================"
  sleep 2
done

echo "ANALYZING DONE!"