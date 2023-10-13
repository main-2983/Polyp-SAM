# PolypSAM Segment Anything for Polyp Segmentation
## Create config
Create file config.py and put in configs folder
## Install requirement
```
git clone https://github.com/main-2983/Polyp-SAM.git
cd segment-anything; pip install -e .
pip install -r requirements.txt
```
## Training guide
```
python interative_training.py config
```
## Testing guide
```
python automatic_val.py checkpoint_path --store_path test_folder_path
```