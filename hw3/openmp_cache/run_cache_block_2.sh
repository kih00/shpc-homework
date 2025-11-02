make

num_threads=$1

for i in {128..960..64}; do
  ./run.sh -v -t $num_threads -n 3 $i $i $i $i
  echo "================================"
  sleep 2
done

for i in {1024..1920..128}; do
  ./run.sh -v -t $num_threads -n 3 $i $i $i $i
  echo "================================"
  sleep 2
done

for i in {2048..3840..256}; do
  ./run.sh -v -t $num_threads -n 3 $i $i $i $i
  echo "================================"
  sleep 2
done

echo "ANALYZING DONE!"