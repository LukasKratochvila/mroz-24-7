# Image segmentation of Floor plan
Python framework for image segmentation.
This repository origin from work Gijs de Jong: [repository](https://github.com/TheOnlyError/2d3d)

Thesis: **[Multi-Unit Floor Plan Recognition and Reconstruction](https://repository.tudelft.nl/islandora/object/uuid%3A158f6745-0b43-4796-b21d-6388a35f5a2d?collection=education)**
<br>
[Gijs de Jong](https://github.com/TheOnlyError)
<br>

Contents:
* [Creation of virtual environment](#creation-of-virtual-environment)
* [Create Mask](#create-mask)
* [Predicting](#predicting)
* [Evaluation](#evaluation)


## Creation of virtual environment
To create and setup venv install `python3.8` and `virtualenv`:

For Linux users:
```bash
cd <project_folder>
python3.8 -m venv venv  # Create virtual environment
source venv/bin/activate  # Virtual environment activation 
pip install --upgrade pip
pip install -r requirements.txt 
```
For Windows users:
```bash
cd <project_folder>
python3.8 -m venv venv  # Create virtual environment
source venv/Scripts/activate.bat  # Virtual environment activation 
pip install --upgrade pip
pip install -r requirements.txt
pip install regex 
```

## Create Mask
To process the pdf file, make sure you have `inkscape` version `1.1.1` installed, if you don't want to use inkscape, create files with `.png` and `.svg`!

For creating data mask (used for training) can be used:
> Make sure you had installed [virtual environment](#creation-of-virtual-environment) and it is active!
```bash
python create_mask.py -p <path-to-dataset>
```
The script assumes you have this structure (folder and file names may vary). Files in quotation marks are created:

    .
    ├── ...
    └── Data                            # Folder with data samples
        └── <1>                         # Folder with one data sample
        │   ├── <file>.pdf              # Source pdf file
        │   └── <file>_farba.pdf        # Annotation pdf file
        "│   ├── <file>.png"            # Source png file - created
        "│   └── <file>_farba.svg"      # Annotation svg file - created
        ├── <2>                         # Folder with second data sample
        │   └── ...
        └── ... 


## Predicting
For predicting, you will need [pretrained model](https://drive.google.com/file/d/1AeYKb1j1jLyZMwfCnZGnSddZ0k4pIY_N/view?usp=drive_link) and images to predict. One can use single image or folder with images.

When you have prepared images and config, you can run predicting by this command:
> Make sure you had installed [virtual environment](#creation-of-virtual-environment) and it is active!

> Make sure that `log_dir` in config file is correct!

> Script can process pdf files. Parameter `w_n` in [predict_config.py](predict_config.py) is used for image resizing due to OS memory consumption when processing pdf. (If you want to process `pdf` make sure that you have installed `poppler`, because pdf2image python module is only wrapper around)
```bash
python predict_config.py <config> <image or image folder>
```

By mistake, the prediction is confused. To correct output from model run script (in this script model class should be same as classes in model config file):
> Make sure you had installed [virtual environment](#creation-of-virtual-environment) and it is active!

```bash
python correct_result.py <result image>
```


## Evaluation
To evaluate dataset prepare folders as follow - create mask file and create result file (correct the result file) and run this script:
> Make sure you had installed [virtual environment](#creation-of-virtual-environment) and it is active!
```bash
python evaluate.py -p <path-to-dataset>
```
The script assumes you have this structure (folder and file names may vary). Files in quotation marks are created:

    .
    ├── ...
    └── Data                            # Folder with data samples
        └── <1>                         # Folder with one data sample
        │   ├── mask.png                # Mask file created with create_mask.py script
        │   └── result_<file>.png       # Result file created with predict_config.py (corrected by correct_result.py)
        ├── <2>                         # Folder with second data sample
        │   └── ...
        └── ... 

# License
[Apache License v2.0](LICENSE)
