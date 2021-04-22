# now, we are going to create the virtual environment ONCE in the login node
# after being created in the login node, it is just activated in job scripts
cd $HOME  # go to /home/varun94
module load python/3.6
echo "(in-job venv setup) Creating virtual environment at $SLURM_TMPDIR/venv for regression U-Nets..."
virtualenv --no-download $HOME/Runets_PL
source $HOME/Runets_PL/bin/activate
# note we DO need to download new pip verison now, so we omit the --no-index here
python -m pip install --upgrade pip
echo "(in-job venv setup) Installing regression U-Net dependencies..."
pip install --no-index tensorflow_gpu
echo "(in-job venv setup) Done installing tensorflow_gpu for regression U-Nets"
pip install --no-index Keras isort matplotlib nibabel pytest scikit-image scikit-learn scipy sklearn tensorboard torch torchio torchvision tqdm tensorboardX
pip install --upgrade pytorch-lightning
echo "(in-job venv setup) Done installing main Python tools for regression U-Nets"
pip install --no-index opencv-python==3.4.4.19
echo "(in-job venv setup) Done installing opencv-python for regression U-Nets"
echo
echo "(in-job venv setup) Done preparing virtual environment for regression U-Nets!"
