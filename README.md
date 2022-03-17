# Pytorch Implementation of the Supervised Grammar Variational Autoencoder Model

This code is a pytorch implementation for the paper: https://doi.org/10.1021/acs.jcim.1c01573 


## Requirements

torch-1.10.0  
skorch-0.11.0  
torchinfo-1.6.3

We use torchinfo to have an output of the model architecture similar to keras' ```model.summary()```. It's not necessary for the code to run.

## Creating the data set

To create the molecular dataset, use:
* ```python make_dataset_grammar.py```

## Training

To train the model, simply run:
* ```python train.py```

All the relevant information for the training procedure, such as number of epochs, batch size, dimension of the latent space and many others, can be set in the ```parameters.py``` file. By default, the results will be saved in a sequence of folders using the following structure:

```
results
└───property_name
│   └───timestamp
│       │   log.csv
│       │   log_val.csv
│       │   gvae_encoder.pth
│       |   ...
|       └───evaluation
|           |   latent_space.png
|           |   metrics.json
|           |   ...
└───property_name
    └───timestamp
    ...
```
```property_name``` refers to the chosen property to train the model with. In case you run it without any property (vanilla GVAE), such folder will be named as ```no_prop```. ```timestamp``` refers to current date and time, and the folder will be name as follows: day_month_year_hour_minutes_seconds. Finally, most of the results of the model, such as the latent space visualization, metrics os prior validity and property prediction performance, will be saved in the ```evaluation``` folder.
