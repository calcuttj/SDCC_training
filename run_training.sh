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
XVU=${XVU:-0}
app="dnnroi_custom"

if [ $XVU -eq 1 ]; then
  app="xvunet"
fi 

if [ $do_amp -eq 1 ]; then
  echo "adding --amp"
  amp_flag="--amp"

  if [ $XVU -eq 1 ]; then
    echo "setting amp type: bfloat16"
    amp_flag="${amp_flag} --amp-dtype bfloat16"
    echo "amp_flag: $amp_flag"
  fi
fi

seed=${seed:-""}
if [ "$seed" != "" ]; then
  echo "Setting seed to $seed"
  seed="--manual-seed ${seed}"
fi

ddp_split_seed=${ddp_split_seed:-""}
if [ "$ddp_split_seed" != "" ]; then
  echo "Setting ddp_split_seed to $ddp_split_seed"
  ddp_split_seed="--ddp-split-seed ${ddp_split_seed}"
fi

load=${load:-""}
if [ "$load" != "" ]; then
  echo "Loading checkpoint file $load"
  load_name=${load}
  load="--load ${load}"
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

base_epochs=0
if [ "$load" != "" ]; then
  base_epochs=$(python -c "import torch; print(len(torch.load('${load_name}')['epochs'])) ")
  this_exit=$?
  if [ $this_exit -ne 0 ]; then
    echo "could not extract base_epochs. exit error $this_exit"
    exit $this_exit
  fi
  echo "Base epochs: $base_epochs"
fi


mkdir -p $workdir && cd $workdir

output_file=training_results_${ProcessId}_${ClusterId}.pt
output_md=training_metadata_${ProcessId}_${ClusterId}.txt


do_ddp=${do_ddp:-0}
ddp_ngpu=${ddp_ngpu:-1}
if [ $do_ddp -eq 1 ]; then
  ddp_devices=${ddp_devices:-0}
  #ddp_ngpu=${ddp_ngpu:-1}
  #torchrun --standalone --nproc_per_node={NProcs} \
  #-m wirecell.dnn

  export NCCL_P2P_DISABLE=1
  export NCCL_IB_DISABLE=1
  #export CUDA_VISIBLE_DEVICES="${ddp_devices}" 
  echo "cuda visible devices: $CUDA_VISIBLE_DEVICES"
  run="torchrun --standalone --nproc_per_node=${ddp_ngpu} -m wirecell.dnn"
else
  run="wcpy dnn"
fi

$run train -e ${epochs} -b ${batch} --eval-batch ${ebatch} -d ${device} \
        -a $app -s ${output_file} -c ${cfg_file} \
        --checkpoint-save checkpoint_${ProcessId}_${ClusterId}_{epoch}.pt \
        ${seed} ${ddp_split_seed} \
        ${load} \
        --checkpoint-modulus ${checkpoint_mod} ${amp_flag} ${cache_flag}
train_exitcode=$?

echo """cfg: ${cfg_file}
epochs: ${epochs}
batch: ${batch}
device: ${device}
output location: ${output_dir}
output: ${output_file}
checkpoints: checkpoint_${ProcessId}_${ClusterId}_{epoch}.pt""" > ${output_md}

if [ $train_exitcode  -ne 0 ]; then
  echo "Training exited with $train_exitcode"
  exit ${train_exitcode}
fi
mkdir -p ${output_dir}
cp ${output_file} ${output_dir}/
cp ${output_md} ${output_dir}/
checkpoints=$(ls checkpoint_*pt)
for f in ${checkpoints[@]}; do cp $f ${output_dir}/; done

mkdir -p metrics/
wcpy dnn viztrain --no-dots -o metrics/loss_curves.png  --mean-train ${output_file}
wcpy dnn viztrain --no-dots -o metrics/loss_curves_logy.png  --mean-train --logy ${output_file}

if [[ XVU -ne 0 ]]; then
  shape="1,1,2560,1500"
  wcpy dnn export-ts -l ${output_file} -o $(echo ${output_file} | sed -e "s/\.pt/\.ts/") --sigmoid -c ${cfg_file}  -m trace --shape ${shape}
fi


DO_METRICS=${DO_METRICS:-0}
extra=""
if [[ XVU -ne 0 ]]; then
  extra="-xvu"
fi

if [[ DO_METRICS -eq 1 ]]; then
  echo "Will do metrics"

  snakefile=${snakefile:-/home/dune/users/jcalcutt/wire-cell-toolkit/spng/test/training_inputs/Snakefile}
  # export MODULEPATH=/home/dune/users/jcalcutt/spack_data/modules/linux-almalinux9-x86_64:/home/dune/users/jcalcutt/spack_data/modules/linux-almalinux9-zen4:$MODULEPATH
  export WIRECELL_PATH=/home/dune/users/jcalcutt/wire-cell-data/:/home/dune/users/jcalcutt/wire-cell-toolkit/cfg/:/home/dune/users/jcalcutt/wire-cell-toolkit/spng/:/home/dune/users/jcalcutt/wire-cell-toolkit/spng/cfg/:$WIRECELL_PATH
  export PATH=/home/dune/users/jcalcutt/training-campaign-spng-cm-sep26/install/bin:$PATH
  ncores_snakemake=${ncores_snakemake:-8}
  metrics_plane=${metrics_plane:-u}


  
  epoch_metrics=${epoch_metrics:-0}
  if [[ XVU -eq 1 ]]; then
    targets="all_eff_pur_xvu"
    if [[ epoch_metrics -eq 1 ]]; then
      targets="${targets} all_epochs_xvu"
    fi
  else
    targets="results/all_eff_pur_${metrics_plane}plane.npz"
    if [[ epoch_metrics -eq 1 ]]; then
      targets="${targets} $(echo $(for i in $(seq $base_epochs $(( base_epochs+epochs-1 ))); do echo epoch_${i}/all_eff_pur_${metrics_plane}plane.npz; done))"
    fi
  fi

  jsonnet_str=""
  if [[ trios -eq 1 ]]; then
    jsonnet_str="mpsigproc_training_js=dnnroi-training-trios.jsonnet"
    targets="${targets} xvu-trio-crossplane-uplane-t1_75-t2_75-g4-trio.npz" ##TODO add more
  fi

  ##make sure module function is available
  # source /etc/profile.d/modules.sh

  ## Make the plots for the final training epoch
  # --use-envmodules -- removed 
  snakemake --snakefile ${snakefile} -d metrics \
        --cores ${ncores_snakemake} \
        --resources gpu=2 \
        --config model=$PWD/${output_file} \
                 cfg=${cfg_file} \
                 cpt_dir=$PWD \
                 proc=${ProcessId} \
                 dnnapp=${app} \
                 cluster=${ClusterId} \
                 device=${device} \
                 ${jsonnet_str} \
        -- $targets #all_eff_pur_xvu all_epochs_xvu
        #-- $(echo $(for mp in $metrics_plane; do echo results/all_eff_pur_${mp}plane${extra}.npz; done)) $(echo $(for i in $(seq 0 $(( epochs-1 ))); do for mp in ${metrics_plane}; do echo epoch_${i}/all_eff_pur_${mp}plane${extra}.npz; done; done))
  result=$?
  if [[ result -ne 0 ]]; then
    echo "Snakemake 1 exited with ${result}"
    exit $result
  fi


  if [[ epoch_metrics -eq 1 ]]; then
    ## Make the per-epoch summary plots
    targets="results/eff_pur_epochs_${metrics_plane}plane.pdf"
    if [[ XVU -eq 1 ]]; then
      targets="results/eff_pur_epochs_uplane-xvu.pdf results/eff_pur_epochs_vplane-xvu.pdf results/eff_pur_epochs_wplane-xvu.pdf"
    fi
    # --use-envmodules 
    snakemake --snakefile ${snakefile} -d metrics \
          --cores ${ncores_snakemake} \
          --resources gpu=2 \
          --config model=$PWD/${output_file} \
                  cfg=${cfg_file} \
                  cpt_dir=$PWD \
                  proc=${ProcessId} \
                  dnnapp=${app} \
                  cluster=${ClusterId} \
                  device=${device} \
          -- $targets #results/eff_pur_epochs_${metrics_plane}plane${extra}.pdf
    result=$?
    if [[ result -ne 0 ]]; then
      echo "Snakemake 2 exited with ${result}"
      exit $result
    fi
  fi
  
  tar -czf metrics/all_rec_tru.tar.gz metrics/*-{rec,tru}*.h5
  rm metrics/*-{rec,tru}*.h5

  threshold_metrics=${threshold_metrics:-0}
  if [[ threshold_metrics -eq 1 ]]; then
    threshold_targets="$(echo aggregated_scan_{roi,pixel}_plane_{u,v,w}_fbetas.png)"
    threshold_snakefile=${threshold_snakefile:-/home/dune/users/jcalcutt/wire-cell-toolkit/spng/test/training_inputs/training_output.snakefile}
    if [[ XVU -eq 1 ]]; then
      threshold_targets="${threshold_targets} $(echo aggregated_scan_{roi,pixel}_plane_all_planes_fbetas.png)"
    fi;
    echo "tresh targs: $threshold_targets"
    test_cosmic_files=${test_cosmic_files:-"/home/dune/users/jcalcutt/spng_cm_sep26_retraining_rebin_fix/20??_0/*cosmics_*-g4-*-*.h5"}
    snakemake --snakefile ${threshold_snakefile} --cores=${ncores_snakemake} \
      --config paths="$(echo ${test_cosmic_files})" \
               model_file="$PWD/${output_file}" \
               cfg=${cfg_file} nentries=200 threshold_step=.025 device=${device} --directory metrics/threshold_scan --resources gpu=6 \
      -- ${threshold_targets}
  fi

fi


cp -r metrics/ ${output_dir}/

du -sh .
echo "Output dir: ${output_dir}"
