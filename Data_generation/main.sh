#!/bin/sh

#  assert correct run dir
run_dir="Bachelor"
if ! [ "$(basename $PWD)" = $run_dir ];
then
    echo -e "\033[0;31mScript must be submitted from the directory: $run_dir\033[0m"
    exit 1
fi

# create dir for logs
mkdir -p "logs/"

### General options
### –- specify queue --
#BSUB -q hpc
### -- set the job Name --
#BSUB -J UNet
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4 
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now 
#BSUB -W 23:59
### -- request 5GB of system-memory --
#BSUB -R "rusage[mem=5GB]"
### -- set the email address --
##BSUB -u s193961@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion-- 
#BSUB -N
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o logs/Unet%J.out 
#BSUB -e logs/Unet%J.err 
### -- end of LSF options --

# activate env
source bachelor_env/bin/activate

# load additional modules
module load scipy/1.7.3-python-3.9.11
module load cuda/11.4

# Which cases to use


#Clears work data folder
cd /
cd work3/s193961/WorkingData/
for files in C*
do
rm -rf $files
done


#Copies relevant folders to work directory
cd /
cd work3/s193961/AllData/
for ca in C*
do 
    #Which cases to use
    allcases="00 01 02 03 04 05 06 07 08 09 10 11"
    #Which Distances to use
    alldistances="08"
    case=${ca:4:2}
    for item in $allcases
    do
    if [ "$case" == "$item" ]; then
        cd /
        cd work3/s193961/WorkingData/
        mkdir $ca
        cd /
        cd work3/s193961/AllData/
        cd $ca

        for w in WT*
        do
        distance=${w:2:2}

        for dist in $alldistances
        do
            if [ "$distance" == "$dist" ]; then
                pwd -P
                cp -r $w /work3/s193961/WorkingData/$ca/
            fi
        done 
        done
    fi
    done
done




cd /
cd work3/s193961/WorkingData/

for cases in C*
do
    cd $cases
    for distances in WT*
    do
        cd $distances
        mkdir winddata
        mv LES* winddata/.
        cp -r /zhome/87/3/146402/Desktop/Bachelor/Flexfiles/base winddata/
        
        for i in y*
        do
            rm -rf $i
        done
        # yaw range for 2 turbine
        for y in {-30..30..2}
        do
            # pitch range for 2 turbine
            for p in {0..0..5}
            do
                # rotspeed range for 2 turbine
                for o in {80..80..5}
                do
                    n1=-1
                    y2=`expr $y \* $n1`
                    if [ $y -ge 10 ]
                    then
                        cp -R winddata/ y0${y}_p0${p}_o${o}/
                    elif [ $y -lt 10 ] && [ $y -ge 0 ]
                    then
                        cp -R winddata/ y00${y}_p0${p}_o${o}/
                    elif [ $y -lt 0 ] && [ $y -gt -10 ]
                    then
                        cp -R winddata/ y-0${y2}_p0${p}_o${o}/
                    elif [ $y -le -10 ]
                    then
                        cp -R winddata/ y-${y2}_p0${p}_o${o}/
                    fi
                done
            done
        done
        cp -a /zhome/87/3/146402/Desktop/Bachelor/Flexfiles/Flex5 .
        cd ../
    done
    cd ../
done



cd /
cd zhome/87/3/146402/Desktop/Bachelor/
# run scripts
python3.9 changeinfill.py



cd /
cd work3/s193961/WorkingData/

for cases in C*
do
    cd $cases
    for distances in WT*
    do
        cd $distances
        for B in y*
            do
                cd $B
                for D in LES* 
                do 
                    cd $D
                    cp ../base/IEA* .
                    cp ../base/Fund* .
                    cp ../base/Infile* .
                    cp ../base/Flex5 .
                    ./Flex5
                    cd ../
                done
                cd ../
            done
        cd ../
    done
    cd ../
done

cd /
cd zhome/87/3/146402/Desktop/Bachelor/OutputFiles/
for files in y*
do
    rm -rf $files
done

cd /
cd zhome/87/3/146402/Desktop/Bachelor/Training/

for i in T*
    do
        rm -rf $i
    done


cd /
cd zhome/87/3/146402/Desktop/Bachelor/OutputFilesDEL2/

for i in DEL*
    do
    cd $i
    for j in y*
        do
            rm -rf $j
        done
    cd ../
    done

cd /
cd zhome/87/3/146402/Desktop/Bachelor/
# run scripts
python3.9 meancalculaterfirst.py
python3.9 weibullfit.py
python3.9 weibullfitDEL.py