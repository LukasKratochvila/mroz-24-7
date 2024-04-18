# Image segmentation of Floor plan
Python framework for image segmentation.
This repository origin from work Gijs de Jong: [repository](https://github.com/TheOnlyError/2d3d)

Thesis: **[Multi-Unit Floor Plan Recognition and Reconstruction](https://repository.tudelft.nl/islandora/object/uuid%3A158f6745-0b43-4796-b21d-6388a35f5a2d?collection=education)**
<br>
[Gijs de Jong](https://github.com/TheOnlyError)
<br>

Contents:
* [Creation of virtual environment](#creation-of-virtual-environment)
* [Predicting](#predicting)

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
```

## Predicting
For predicting, you will need [pretrained model](https://drive.google.com/file/d/1AeYKb1j1jLyZMwfCnZGnSddZ0k4pIY_N/view?usp=drive_link) and images to predict. One can use single image or folder with images.

When you have prepared images and config, you can run predicting by this command:
> Make sure you had installed [virtual environment](#creation-of-virtual-environment) and it is active!

> Make sure that `log_dir` in config file is correct!

> Script can process pdf files. Parameter `w_n` in [predict_config.py](predict_config.py) is used for image resizing due to OS memory consumption when processing pdf. 
```bash
python predict_config.py <config> <image or image folder>
```


# License
[Apache License v2.0](LICENSE)
