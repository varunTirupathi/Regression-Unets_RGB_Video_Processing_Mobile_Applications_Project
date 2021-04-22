#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --gres=gpu:v100:4
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=12G      # increase as needed
#SBATCH --time=2-05:00:00         # increase if longer runtimes are needed
#SBATCH --output=%N-%j.out
#SBATCH --mail-user=varun.tirupathi@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

PROJECT="$HOME/Regression-Unets-RGB-video-processing-"
IMAGES="$HOME/dataset/images"

# now, we are going to create the virtual environment ONCE in the login node
# after being created in the login node, it is just activated in job scripts
cd $SLURM_TMPDIR  # go to /home/varun94
module load python/3.6
source $HOME/Runets_venv/bin/activate
echo 'Successfully loaded $HOME virtual environment'
python $PROJECT/main.py --mask_filter "CannyEdgeDetection" --img_dir $IMAGES
