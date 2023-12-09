echo "python run.py --epoches 140000 --modelLoad AA --weightModel identical --randomGraph BA --seed 14 --strains 2 --CMDprogress_78_79"
python run.py --epoches 140000 --modelLoad AA --weightModel identical --randomGraph BA --seed 14 --strains 2 --CMDprogress_78_79 &
echo "python run.py --epoches 140000 --modelLoad AA --weightModel identical --randomGraph BA --seed 14 --strains 3 --CMDprogress_79_79"
python run.py --epoches 140000 --modelLoad AA --weightModel identical --randomGraph BA --seed 14 --strains 3 --CMDprogress_79_79 &
wait

python run.py --epoches 140000 --modelLoad AA --weightModel identical --randomGraph BA --seed 14 --strains 3 --CMDprogress_79_79 &

