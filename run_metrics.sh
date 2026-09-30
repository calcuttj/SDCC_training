#!/bin/bash
# Need definition:
# - checkpoint -- training output
# - cfg_file -- cfg file used for training

if [[ -z "${checkpoint}" ]]; then
    echo "Need to specify checkpoint"
    exit 1
fi
if [[ -z "${cfg_file}" ]]; then
    echo "Need to specify cfg_file"
    exit 1
fi


ProcessId=${ProcessId:-0}
ClusterId=${ClusterId:-0}
workdir=/tmp/jcalcutt/metricsdir_${ProcessId}_${ClusterId}/
output_top=${output_top:-/gpfs/mnt/gpfs01/lbne/users/spng/jcalcutt/training_metrics/}
extra_dir_name=${extra_dir_name:-""}
output_dir=${output_top}/${ProcessId}_${ClusterId}
if [ -n "${extra_dir_name}" ]; then
  output_dir="${output_dir}_${extra_dir_name}"
fi

mkdir -p $workdir && cd $workdir

## Set up venv 
## This includes both wcpy and snakemake now
source /home/dune/users/jcalcutt/wire-cell-python/.venv/bin/activate


snakefile=${snakefile:-/home/dune/users/jcalcutt/wire-cell-toolkit/spng/test/training_inputs/Snakefile}
threshold_snakefile=${threshold_snakefile:-/home/dune/users/jcalcutt/wire-cell-toolkit/spng/test/training_inputs/training_output.snakefile}
export WIRECELL_PATH=/home/dune/users/jcalcutt/wire-cell-data/:/home/dune/users/jcalcutt/wire-cell-toolkit/cfg/:/home/dune/users/jcalcutt/wire-cell-toolkit/spng/:/home/dune/users/jcalcutt/wire-cell-toolkit/spng/cfg/:$WIRECELL_PATH
export PATH=/home/dune/users/jcalcutt/training-campaign-spng-cm-sep26/install/bin:$PATH
ncores_snakemake=${ncores_snakemake:-8}
metrics_plane=${metrics_plane:-u}
device=${device:-cpu}
XVU=${XVU:-0}
app="dnnroi_custom"

if [ $XVU -eq 1 ]; then
  app="xvunet"
fi 
#increase soft limit of n files 
ulimit -Sn 10240 

# Flow: 
# - Take the linedepos from a 

threshold_targets="$(echo aggregated_scan_{roi,pixel}_plane_{u,v,w}_fbetas.png)"
if [[ XVU -eq 1 ]]; then
    threshold_targets="${threshold_targets} $(echo aggregated_scan_{roi,pixel}_plane_all_planes_fbetas.png)"
fi

test_cosmic_files=${test_cosmic_files:-"/home/dune/users/jcalcutt/spng_cm_sep26_retraining_rebin_fix_trios/11??_0/*cosmics_*-g4-*-*.h5"}
snakemake --snakefile ${threshold_snakefile} --cores=${ncores_snakemake} \
    --config paths="$(echo ${test_cosmic_files})" \
            model_file="${checkpoint}" \
            cfg=${cfg_file} nentries=10 threshold_step=.025 device=${device} --directory metrics/threshold_scan --resources gpu=6 \
    -- ${threshold_targets}

linedepos_loc=${linedepos_loc:-"/home/dune/users/jcalcutt/spng_cm_sep26_retraining_rebin_fix_trios_linedepos/test"}
# for f in "${linedepos_loc}/*h5"; do
    # echo $f
    ln -s ${linedepos_loc}/*h5 metrics/
# done

targets="all_eff_pur_xvu"
snakemake --snakefile ${snakefile} -d metrics \
    --cores ${ncores_snakemake} \
    --resources gpu=2 \
    --config model=${checkpoint} \
                cfg=${cfg_file} \
                dnnapp=${app} \
                device=${device} \
    -- $targets


targets=""
touch threshold_scan_info.txt
for type in roi pixel; do
  for beta in 1.0 0.5; do
    threshold=$(python -c "import numpy as np; t = np.load('metrics/threshold_scan/fbeta_aggregated_scan_${type}_plane_all_planes_${beta}.npz'); maxloc = t['maxloc']; print(f'{maxloc.item():.3f}')")
    echo "${type} ${beta} ${threshold} all_planes" >> threshold_scan_info.txt
    for p in u v w; do
        targets+="results/threshold-${threshold}-xvu-all_${type}_eff_pur_${p}plane.png results/threshold-${threshold}-xvu-trio-merged-crossplane_${p}plane_high_end.png "

        threshold=$(python -c "import numpy as np; t = np.load('metrics/threshold_scan/fbeta_aggregated_scan_${type}_plane_${p}_${beta}.npz'); maxloc = t['maxloc']; print(f'{maxloc.item():.3f}')")
        echo "${type} ${beta} ${threshold} ${p}" >> threshold_scan_info.txt
        targets+="results/threshold-${threshold}-xvu-all_${type}_eff_pur_${p}plane.png results/threshold-${threshold}-xvu-trio-merged-crossplane_${p}plane_high_end.png "
    done
  done
done
echo $targets
# exit
# targets="results/threshold-0.5-xvu-all_roi_eff_pur_uplane.png"
snakemake --snakefile ${threshold_snakefile} -d metrics \
    --cores ${ncores_snakemake} \
    --resources gpu=2 \
    -- $targets
    # --config device=${device} \

echo "Moving outputs to ${output_dir}"
mkdir -p ${output_dir}
echo "checkpoint=${checkpoint}" >> ${output_dir}/metrics_metadata.txt
echo "cfg=${cfg_file}" >> ${output_dir}/metrics_metadata.txt

cp -r metrics/ ${output_dir}/
cp threshold_scan_info.txt ${output_dir}/