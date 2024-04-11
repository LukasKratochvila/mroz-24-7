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
For predicting, you will need [pretrained model]()

When you have prepared data and config, you can run training by this command:
> Make sure you had installed [virtual environment](#creation-of-virtual-environment) and it is active
```bash
python predict_config.py
```


# License
[Apache License v2.0](LICENSE)
