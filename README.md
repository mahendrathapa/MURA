# This repo dedicated for the research using Standford's Musculoskeletal Radiographs (MURA) dataset to detect Abnormality #

# Train Model #
``` python main.py --env server --model train ```

# Inference from Model #
``` python main.py --env server --mode predict --run_id <unique identifier> --model_checkpoint <saved model present inside out/<run_id>/checkpoints --predict_data_dir <image dir for inference>```

### Set up Conda environment ###
``` bash setup.sh ```

### Activate Conda environment ###
``` conda activate mura ```

# Run the Flask Server #
Flask servers the trained model through API, Before serving, we need to add the RUN_ID and MODEL_NAME in src/constant.py
``` python -m src.api ```

# Front-End #
Front-End App is developed in React.

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