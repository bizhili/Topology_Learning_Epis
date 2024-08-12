echo "python run.py --epoches 10000 --modelLoad AB --weightModel identical --randomGraph us_air300        --seed 11 --strains 4  --dense 0 --n 100 --intense -1 --CMDprogress_0_4"
python run.py --epoches 10000 --modelLoad AB --weightModel identical --randomGraph us_air300        --seed 11 --strains 4  --dense 0 --n 100 --intense -1 --CMDprogress_0_4 &
echo "python run.py --epoches 10000 --modelLoad AB --weightModel identical --randomGraph global_air        --seed 11 --strains 4  --dense 0 --n 100 --intense -1 --CMDprogress_1_4"
python run.py --epoches 10000 --modelLoad AB --weightModel identical --randomGraph global_air        --seed 11 --strains 4  --dense 0 --n 100 --intense -1 --CMDprogress_1_4 &
echo "python run.py --epoches 10000 --modelLoad AB --weightModel identical --randomGraph us_air400        --seed 11 --strains 4  --dense 0 --n 100 --intense -1 --CMDprogress_2_4"
python run.py --epoches 10000 --modelLoad AB --weightModel identical --randomGraph us_air400        --seed 11 --strains 4  --dense 0 --n 100 --intense -1 --CMDprogress_2_4 &
echo "python run.py --epoches 10000 --modelLoad AB --weightModel identical --randomGraph us_air200        --seed 11 --strains 4  --dense 0 --n 100 --intense -1 --CMDprogress_3_4"
python run.py --epoches 10000 --modelLoad AB --weightModel identical --randomGraph us_air200        --seed 11 --strains 4  --dense 0 --n 100 --intense -1 --CMDprogress_3_4 &
wait
echo "python run.py --epoches 10000 --modelLoad AB --weightModel identical --randomGraph mobility_german        --seed 11 --strains 4  --dense 0 --n 100 --intense -1 --CMDprogress_4_4"
python run.py --epoches 10000 --modelLoad AB --weightModel identical --randomGraph mobility_german        --seed 11 --strains 4  --dense 0 --n 100 --intense -1 --CMDprogress_4_4 &
wait
