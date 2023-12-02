conda activate tf

python test2.py &
python test2.py &
wait
python test2.py --n 1 &
python test2.py --n 2 &

wait