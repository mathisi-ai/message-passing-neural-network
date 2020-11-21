**GENERAL PARAMETERS**

- Your dataset folder is defined by: 

DATASET_NAME='sample-dataset'

- Your dataset directory is defined by: 

DATA_DIRECTORY='data/'

- The directory to save the model checkpoints is defined by: 

MODEL_DIRECTORY='model_checkpoints'

- The directory to save the grid search results per configuration is defined by: 

RESULTS_DIRECTORY='grid_search_results'

- The option to run the model on 'cpu' or 'cuda' can be controlled by: 

DEVICE='cpu'

**USED FOR GRID SEARCH**

To define a range for the grid search please pass the following values as lists (same for no grid search)

- The model to use (only 'RNN' available at this version') is defined by :

MODEL=['RNN']

- The total number of epochs can be controlled by:

EPOCHS=[10]

- The choice of the loss function can be controlled by (see message_passing_nn/utils/loss_functions.py for a full list):

LOSS_FUNCTION=['MSE']

- The choice of the optimizer can be controlled by (see message_passing_nn/utils/optimizers.py for a full list):

OPTIMIZER=['SGD']

- The batch size can be controlled by:

BATCH_SIZE=[1]

- The validation split can be controlled by:

VALIDATION_SPLIT=[0.2]

- The test split can be controlled by:

TEST_SPLIT=[0.1]

- The message passing time steps can be controlled by:

TIME_STEPS=[5]

- The number of epochs to evaluate the model on the validation set can be controlled by:

VALIDATION_PERIOD=[5]

**USED FOR INFERENCE**

- The model to load (only 'RNN' available at this version') is defined by :

MODEL='RNN'

Please don't pass the inference model as a list!