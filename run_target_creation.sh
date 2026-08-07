#!/bin/bash 

Start=${Start:-0}
ProcessId=${ProcessId:-0}
ClusterId=${ClusterId:-0}
NEVENTS=${NEVENTS:-10}
FileIndex=$(( Start+ProcessId ))
Device=${Device:-"cpu"}
echo $Start $FileIndex $ClusterId $Device
workdir=/tmp/jcalcutt/workdir_${ProcessId}_${ClusterId}/
echo "making dir"
mkdir -p $workdir
cd $workdir
echo "PWD: $PWD"
target=${target:-"electron"}

#JUSTIN_JOBSUB_ID=${ProcessId}_${ClusterId}
#JUSTIN_STAGE_ID=1

#Set this for jobscript to point to fcls etc.
export input_path=/lbne/u/jcalcutt/training_condor/cosmics_g4_depos_dir/

## Run dnnroi script in apptainer SL7
## TODO Full list of binds
echo "Running"
apptainer exec -B /cvmfs,/gpfs01 \
        --env input_path=${input_path} \
        --env JUSTIN_PATH=/lbne/u/jcalcutt \
        --env target=${target} \
        --env script_path=/lbne/u/jcalcutt/training_condor/scripts/ \
        --ipc --pid  /cvmfs/singularity.opensciencegrid.org/fermilab/fnal-dev-sl7:latest /lbne/u/jcalcutt/training_condor/dnnroi_training_various_depos.jobscript



source /home/dune/users/jcalcutt/wire-cell-python/.venv/bin/activate
source /etc/profile.d/modules.sh
export MODULEPATH=/home/dune/users/jcalcutt/spack_data/modules/linux-almalinux9-x86_64:/home/dune/users/jcalcutt/spack_data/modules/linux-almalinux9-zen4:$MODULEPATH

export LD_LIBRARY_PATH=/lbne/u/jcalcutt/spack_installs/linux-x86_64/wire-cell-toolkit-spng-lspm56haxualtduash6pkzzuxklub773/lib/:$LD_LIBRARY_PATH

export WIRECELL_PATH=/home/dune/users/jcalcutt/wire-cell-data/:/home/dune/users/jcalcutt/wire-cell-toolkit/cfg/:/home/dune/users/jcalcutt/wire-cell-toolkit/spng/:/home/dune/users/jcalcutt/wire-cell-toolkit/spng/cfg/:$WIRECELL_PATH
snakefile=${snakefile:-/home/dune/users/jcalcutt/wire-cell-toolkit/spng/test/training_inputs/Snakefile}
sample="1_1"
snakemake --use-envmodules --snakefile ${snakefile} --config device=${Device} --cores 1 -- target-${target}_${sample}-g4-rec-{0,1,2}.h5

output_dir=/home/dune/users/jcalcutt/test_${target}_files/${FileIndex}_0

if [ $? -eq 0 ]; then
  echo "Success. Cleaning up"
  #rm $target_dir/$input_filename
  echo "Moving output to ${output_dir}"
  mkdir -p ${output_dir} 
  cp target-${target}_${sample}-g4-rec-0.h5 ${output_dir}
  cp target-${target}_${sample}-g4-rec-1.h5 ${output_dir}
  cp target-${target}_${sample}-g4-rec-2.h5 ${output_dir}
  cp target-${target}_${sample}-g4-tru-0.h5 ${output_dir}
  cp target-${target}_${sample}-g4-tru-1.h5 ${output_dir}
  cp target-${target}_${sample}-g4-tru-2.h5 ${output_dir}

  if [ "${target}" == "electron" ]; then
    cp target-${target}_${sample}_angles.npy ${output_dir}
  fi
fi
