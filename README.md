# latent-transfer

In order to install dependencies, run the following (Note: Mujoco package requires license):
```
pip install -r requirements.txt
```

In order to run a metalearning experiment, run:
```
mkdir ./output
python latent.py --root_dir "./output" --direc
```
to metatrain on forward and backward running, or run
```
mkdir ./output
python latent.py --root_dir "./output" 
```
to metatrain on different velocities for half cheetah

Then to metatest, run
```
python latent.py --root_dir "./output" --direc --finetune
```
or 
```
python latent.py --root_dir "./output" --finetune
```
