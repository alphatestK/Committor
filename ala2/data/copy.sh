for i in {1..2};
do 
    cp ../../../../pic/ala/plumeddriver/iter_1_3d/lambda_$i/I/COLVAR COLVAR.0.$i.R
    cp ../../../../pic/dasa/sims/plumeddriver/iter_1_3d/lambda_$i/P/COLVAR COLVAR.0.$i.P
#cp ../../../../pic/dasa/sims/iter_${i}_new/lambda_2/I2/COLVAR COLVAR.${i}.R
#cp ../../../../pic/dasa/sims/iter_${i}_new/lambda_2/P2/COLVAR COLVAR.${i}.P
done;
#cp ../../../../pic/dasa/sims/iter_0_new/lambda_7/I/COLVAR COLVAR.0.7.R
#cp ../../../../pic/dasa/sims/iter_0_new/lambda_7/P/COLVAR COLVAR.0.7.P

#cp ../../../../pic/dasa/sims/plumeddriver/unbiased_sims/I/data/COLVAR COLVAR.unbiased.R
#cp ../../../../pic/dasa/sims/plumeddriver/unbiased_sims/P/data/COLVAR COLVAR.unbiased.P
