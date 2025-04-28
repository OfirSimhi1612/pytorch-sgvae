# Pytorch Implementation of the Semi-Supervised Grammar Variational Autoencoder Model

## Requirements

torch-1.10.0  
skorch-0.11.0  
torchinfo-1.6.3

We use torchinfo to have an output of the model's architecture similar to keras' `model.summary()`. It's not necessary for the code to run.

## Creating the data set

To create the molecular dataset, use:

- `python make_dataset_grammar.py`

## Training

To train the model, simply run:

- `python train.py`

All the relevant information for the training procedure, such as number of epochs, batch size, dimension of the latent space, percentage of labeled data and many others, can be set in the `parameters.py` file. By default, the results will be saved in a sequence of folders using the following structure:

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

`property_name` refers to the chosen property to train the model with. In case you run it without any property (vanilla GVAE), such folder will be named as `no_prop`. `timestamp` refers to current date and time, and the folder will be name as follows: day_month_year_hour_minutes_seconds. Finally, most of the results of the model, such as the latent space visualization, metrics on prior validity and property prediction performance, will be saved in the `evaluation` folder.

## Testing

To test the model, run:

- `python testing.py --arguments`

The `arguments` that can be passed to the prompt are:  
`--path` (str): path to the file. Ex: --path='results/energy_of_LUMO/16_03_2022_23_4_11'  
`--plot` (store true): plot the two first components of a PCA to visualize the latent space configuration. The TSNE is also set to be plotted.  
`--evaluation` (store true): use this if you want to calculate the prior validity, percentage of novel molecules and percentage of unique molecules.  
`--train_property` (store true): train and test the property prediction model. By default, the model will be trained and tested 5 times, and the final results will be averaged.  
`--hyper_optim` (store true): use this if you want to perform a grid search over the parameters of the property prediction model. We use the **scorch** module to perform the grid search. By default, only 50% of the training data is used in the search. You can change this and set novel hyperparameters to be searched within the `testing.py` file. Be aware that the more data and hyperparameters are used, the longer the searching process will take.  
`--reconstruction` (store true): use this to test the reconstruction accuracy of the model.

## Example of result

You can check an example of result obtained with the model trained with the HOMO energy property in the folder `example`.
