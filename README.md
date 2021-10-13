Hello, and welcome to my Intelinair Segmentation Challenge Repo!

The goal of the task given is to segment the images of fields into two categories, nutrient-deficient and not-nutrient-deficient.

For my model, I'm replicating a slimmed-down version of the state-of-the-art model described in the paper using Keras, with a few minor tweaks to activation functions etc.

Some questions we can answer by doing this are:
-How much does the number of filters matter? Can we see comparable performance with only a few?
-What is the benefit of the pretraining performed in the paper? 

Some potential improvements I could make in the future with more time are:
-Adding an axial self-attention layer to the LSTM module. The recently released METNET (https://arxiv.org/abs/2003.12140) model uses this and sees major gains. 
However, axial self-attention isn't yet supported in keras as far as I know, so this would take some tinkering (but it is in pytorch!).
-In preliminary studies, I saw surprisingly good performance using categorical-cross-entropy loss. It may be useful to consider this as a threefold segmentation problem,
where the classes learned are ("field, not-deficient", "field, deficient", and "not-field (out-of-bounds)").

To handle the boundaries, I've applied the boundary masks manually to the inputs, as well as the outputs of both the U-Nets and the LSTM.
This has the effect of telling the model to ignore anything outside the boundary, since both the masked output and the target are zeroed in these regions.
(There may be a better way to do this, but this one works for now!)

WHAT EACH SCRIPT DOES

config.py holds my global variables, such as image size and location. I also point to a directory containing a single test field for inferencing- 
if you'd like to run this on different test data, you will need to change the config!

get_data.py constructs the imagedatagenerators and flows them from a dataframe. I use horizontal and vertical flip augmentation in the train set to make the model more robust. 

losses.py contains an implementation of the hybrid focal + tversky loss from the paper.

model.py is where I construct the model.

train_model.py is where I train the model. The output is saved in model.h5. Train and validation loss are given for the purpose of hyperparameter tuning and overfit checking.
Loss and IOU curves are stored in output_plots.

run_model_on_test_set.py is where I evaluate the model on the full test set. I also produce the test set output masks, which are stored in output_plots.

inferencing.py is where I set up the model to evaluate all examples in the inferencing directory specified in the config.
The directory I point to is a local one containing a single field from my test set. If you'd like to run this model live, you will need to edit the config to point to your inference data.
