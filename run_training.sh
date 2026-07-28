#!/bin/bash
ProcessId=${ProcessId:-0}
ClusterId=${ClusterId:-0}
workdir=/tmp/jcalcutt/trainingdir_${ProcessId}_${ClusterId}/
output_top=${output_top:-/gpfs/mnt/gpfs01/lbne/users/spng/jcalcutt/training_results}
extra_dir_name=${extra_dir_name:-""}
output_dir=${output_top}/${ProcessId}_${ClusterId}
if [ -n "${extra_dir_name}" ]; then
  output_dir="${output_dir}_${extra_dir_name}"
fi

#configure training
cfg_file=${cfg_file:-/home/dune/users/jcalcutt/wire-cell-python/wirecell/dnn/cfg/spng_campaign/jcal-dnnroi-spng-thresh_mpfile_uplane-pdhd.cfg}
echo "cfg_file"
device=${device:-cpu}
epochs=${epochs:-1}
batch=${batch:-1}
ebatch=${ebatch:-1}
checkpoint_mod=${checkpoint_mod:-1}
do_amp=${do_amp:-0}
do_cache=${do_cache:-0}
amp_flag=""
if [ $do_amp -eq 1 ]; then
  echo "adding --amp"
  amp_flag="--amp"
fi

seed=""
if [ "$seed" != "" ]; then
  echo "Setting seed to $seed"
  seed="--manual-seed ${seed}"
fi

cache_flag=""
if [ $do_cache -eq 1 ]; then
  echo "adding --cache"
  cache_flag="--cache"
fi

#increase soft limit of n files 
ulimit -Sn 10240 

## Set up venv 
## This includes both wcpy and snakemake now
source /home/dune/users/jcalcutt/wire-cell-python/.venv/bin/activate

mkdir -p $workdir && cd $workdir

output_file=training_results_${ProcessId}_${ClusterId}.pt
output_md=training_metadata_${ProcessId}_${ClusterId}.txt
wcpy dnn train -e ${epochs} -b ${batch} --eval-batch ${ebatch} -d ${device} \
        -a dnnroi_custom -s ${output_file} -c ${cfg_file} \
        --checkpoint-save checkpoint_${ProcessId}_${ClusterId}_{epoch}.pt \
        ${seed} \
        --checkpoint-modulus ${checkpoint_mod} ${amp_flag} ${cache_flag}

echo """cfg: ${cfg_file}
epochs: ${epochs}
batch: ${batch}
device: ${device}
output location: ${output_dir}
output: ${output_file}
checkpoints: checkpoint_${ProcessId}_${ClusterId}_{epoch}.pt""" > ${output_md}

mkdir -p ${output_dir}
cp ${output_file} ${output_dir}/
cp ${output_md} ${output_dir}/
checkpoints=$(ls checkpoint_*pt)
for f in ${checkpoints[@]}; do cp $f ${output_dir}/; done

mkdir -p metrics/
wcpy dnn viztrain --no-dots -o metrics/loss_curves.png  --mean-train ${output_file}
wcpy dnn viztrain --no-dots -o metrics/loss_curves_logy.png  --mean-train --logy ${output_file}


DO_METRICS=${DO_METRICS:-0}
if [[ DO_METRICS -eq 1 ]]; then
  echo "Will do metrics"

  snakefile=${snakefile:-/home/dune/users/jcalcutt/wire-cell-toolkit/spng/test/training_inputs/Snakefile}
  export MODULEPATH=/home/dune/users/jcalcutt/spack_data/modules/linux-almalinux9-x86_64:/home/dune/users/jcalcutt/spack_data/modules/linux-almalinux9-zen4:$MODULEPATH
  export WIRECELL_PATH=/home/dune/users/jcalcutt/wire-cell-data/:/home/dune/users/jcalcutt/wire-cell-toolkit/cfg/:/home/dune/users/jcalcutt/wire-cell-toolkit/spng/:/home/dune/users/jcalcutt/wire-cell-toolkit/spng/cfg/:$WIRECELL_PATH

  ncores_snakemake=${ncores_snakemake:-8}
  metrics_plane=${metrics_plane:-u}

  ##make sure module function is available
  source /etc/profile.d/modules.sh

  ## Make the plots for the final training epoch
  snakemake --snakefile ${snakefile} -d metrics \
        --use-envmodules --cores ${ncores_snakemake} \
        --config model=$PWD/${output_file} \
                 cfg=${cfg_file} \
                 cpt_dir=$PWD \
                 proc=${ProcessId} \
                 cluster=${ClusterId} \
                 device=${device} \
        -- results/all_eff_pur_${metrics_plane}plane.npz $(echo $(for i in $(seq 0 $(( epochs-1 ))); do echo epoch_${i}/all_eff_pur_${metrics_plane}plane.npz; done))
  result=$?
  if [[ result -ne 0 ]]; then
    echo "Snakemake 1 exited with ${result}"
    exit $result
  fi
  ## Make the per-epoch summary plots
  snakemake --snakefile ${snakefile} -d metrics \
        --use-envmodules --cores 1 \
        --config model=$PWD/${output_file} \
                 cfg=${cfg_file} \
                 cpt_dir=$PWD \
                 proc=${ProcessId} \
                 cluster=${ClusterId} \
                 device=${device} \
        -- results/eff_pur_epochs_${metrics_plane}plane.pdf
  result=$?
  if [[ result -ne 0 ]]; then
    echo "Snakemake 2 exited with ${result}"
    exit $result
  fi



fi
cp -r metrics/ ${output_dir}/

du -sh .
echo "Output dir: ${output_dir}"
