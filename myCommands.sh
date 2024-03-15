echo "python run.py --epoches 150000 --modelLoad AA --weightModel none --randomGraph us_air        --seed 11 --strains 2  --dense 0 --n 100 --intense -1 --CMDprogress_0_7"
python run.py --epoches 150000 --modelLoad AA --weightModel none --randomGraph us_air        --seed 11 --strains 2  --dense 0 --n 100 --intense -1 --CMDprogress_0_7 &
echo "python run.py --epoches 150000 --modelLoad AA --weightModel none --randomGraph global_air        --seed 11 --strains 4  --dense 0 --n 100 --intense -1 --CMDprogress_1_7"
python run.py --epoches 150000 --modelLoad AA --weightModel none --randomGraph global_air        --seed 11 --strains 4  --dense 0 --n 100 --intense -1 --CMDprogress_1_7 &
echo "python run.py --epoches 150000 --modelLoad AA --weightModel none --randomGraph us_air        --seed 11 --strains 3  --dense 0 --n 100 --intense -1 --CMDprogress_2_7"
python run.py --epoches 150000 --modelLoad AA --weightModel none --randomGraph us_air        --seed 11 --strains 3  --dense 0 --n 100 --intense -1 --CMDprogress_2_7 &
echo "python run.py --epoches 150000 --modelLoad AA --weightModel none --randomGraph us_air        --seed 11 --strains 4  --dense 0 --n 100 --intense -1 --CMDprogress_3_7"
python run.py --epoches 150000 --modelLoad AA --weightModel none --randomGraph us_air        --seed 11 --strains 4  --dense 0 --n 100 --intense -1 --CMDprogress_3_7 &
wait
echo "python run.py --epoches 150000 --modelLoad AA --weightModel none --randomGraph global_air        --seed 11 --strains 2  --dense 0 --n 100 --intense -1 --CMDprogress_4_7"
python run.py --epoches 150000 --modelLoad AA --weightModel none --randomGraph global_air        --seed 11 --strains 2  --dense 0 --n 100 --intense -1 --CMDprogress_4_7 &
echo "python run.py --epoches 150000 --modelLoad AA --weightModel none --randomGraph us_air        --seed 11 --strains 1  --dense 0 --n 100 --intense -1 --CMDprogress_5_7"
python run.py --epoches 150000 --modelLoad AA --weightModel none --randomGraph us_air        --seed 11 --strains 1  --dense 0 --n 100 --intense -1 --CMDprogress_5_7 &
echo "python run.py --epoches 150000 --modelLoad AA --weightModel none --randomGraph global_air        --seed 11 --strains 3  --dense 0 --n 100 --intense -1 --CMDprogress_6_7"
python run.py --epoches 150000 --modelLoad AA --weightModel none --randomGraph global_air        --seed 11 --strains 3  --dense 0 --n 100 --intense -1 --CMDprogress_6_7 &
echo "python run.py --epoches 150000 --modelLoad AA --weightModel none --randomGraph global_air        --seed 11 --strains 1  --dense 0 --n 100 --intense -1 --CMDprogress_7_7"
python run.py --epoches 150000 --modelLoad AA --weightModel none --randomGraph global_air        --seed 11 --strains 1  --dense 0 --n 100 --intense -1 --CMDprogress_7_7 &
wait
