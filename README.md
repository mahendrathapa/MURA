# This repo is dedicated for our attempt in research using Standford's Musculoskeletal Radiographs (MURA) dataset to detect abnormality #

# Set up conda environment #
``` bash setup.sh ```

### Activate conda environment ###
``` conda activate mura ```

# Train Model #
``` python main.py --env server --mode train ```

# Inference from Model #
``` python main.py --env server --mode predict --run_id <unique identifier> --model_checkpoint <saved model present inside out/<run_id>/checkpoints> --predict_data_dir <image dir for inference>```

# Back-End: Run the Flask Server #
Flask serves the trained model through API, before serving, we need to update the RUN_ID and MODEL_NAME in src/constant.py

``` python -m src.api ```

# Front-End #
Front-End App is developed in React.

To host the front end, change to the dir:

``` cd src/frontend ```

## Install React Dependency ##
``` yarn ```

## Run Front-End ##
``` yarn start ```

# Citation #

``` 
@ARTICLE{2017arXiv171206957R,
       author = {{Rajpurkar}, Pranav and {Irvin}, Jeremy and {Bagul}, Aarti and
         {Ding}, Daisy and {Duan}, Tony and {Mehta}, Hershel and {Yang}, Brand
        on and {Zhu}, Kaylie and {Laird}, Dillon and {Ball}, Robyn L. and
         {Langlotz}, Curtis and {Shpanskaya}, Katie and {Lungren}, Matthew P. and
         {Ng}, Andrew Y.},
        title = "{MURA: Large Dataset for Abnormality Detection in Musculoskeletal Radiographs}",
      journal = {arXiv e-prints},
     keywords = {Physics - Medical Physics, Computer Science - Artificial Intelligence},
         year = "2017",
        month = "Dec",
          eid = {arXiv:1712.06957},
        pages = {arXiv:1712.06957},
archivePrefix = {arXiv},
       eprint = {1712.06957},
 primaryClass = {physics.med-ph},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2017arXiv171206957R},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

```
