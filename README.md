Hello, and welcome to my Intelinair Semantic Segmentation Challenge Repo!

The goal of the task given is to segment the images of fields into two categories, nutrient-deficient and not-nutrient-deficient.

For my model, I'm replicating a slimmed-down version of the state-of-the-art model described in the paper using Keras, with a few minor tweaks to activation functions etc.
Some questions we can explore using this architecture are:
-How much does the number of filters matter? Can we see comparable performance with only a few?
-What is the benefit of the pretraining performed in the paper?
-What happens if we tune the lambda, alpha, and beta in the loss?

Some potential improvements I could make in the future with more time are:
-Adding an axial self-attention layer to the LSTM module. The recently released METNET (https://arxiv.org/abs/2003.12140) model uses this and sees major gains. 
However, axial self-attention isn't yet supported in keras as far as I know, so this would take some tinkering (but it is in pytorch!).
-In preliminary studies, I saw surprisingly good performance without applying masks and using vanilla categorical-cross-entropy loss.
It may be interesting to consider this as a threefold segmentation problem, where the classes learned are ("field, not-deficient", "field, deficient", and "not-field (out-of-bounds)").

To handle the boundaries, I've applied the boundary masks manually to the inputs, as well as the outputs of both the U-Nets and the LSTM.
This has the effect of telling the model to ignore anything outside the boundary, since both the masked output and the target are zeroed in these regions.
(There may be a better way to do this, but this one works for now!)
In test where I didn't do this, a lot of the model's energy went toward learning the field boundaries, which isn't useful for this task.

We can see that the model doesn't perform as well as it ought to: it's kind of able to highlight horizontal and vertical strips of field which might be more deficient than others,
but it tends to produce the highest scores near edges and boundaries, and IOU is relatively flat through time. This is a big drawback to not having enough filters!

TO INSTALL AND RUN:

First, download the data from https://registry.opendata.aws/intelinair_longitudinal_nutrient_deficiency/.
Second, if you'd like to retrain the model, edit src/config.py to point to the Longitudinal_Nutrient_Deficiency subdirectory of the directory you have downloaded the data to.
Third, place any new fields you'd like to run inferencing on in data/inference_fields, or edit config.py to point to the directory. 
Fields need to be formatted exactly as they were in the train set: five image files per field, each consisting of a three-channel 256x256 image.
Finally, pip3 install the directory. Possible executables are train_model, run_model_on_test_set, or inferencing.

WHAT EACH SCRIPT DOES

config.py holds my global variables, such as image size and location. I also point to a directory containing a single test field for inferencing- 
if you'd like to run this on test data located somewhere else, you will need to change the config!

get_data.py constructs the imagedatagenerators and flows them from a dataframe. I use horizontal and vertical flip augmentation in the train set to make the model more robust. 
losses.py contains an implementation of the hybrid focal + tversky loss from the paper.
model.py is where I construct the model. It is broken into a few stages: three U-Nets and a ConvLSTM.

train_model.py is where I train the model. The model with the lowest validation loss is saved in model.h5.
Train and validation loss are given for the purpose of hyperparameter tuning and overfit checking.
Loss and IOU curves are stored in output_plots.

run_model_on_test_set.py is where I evaluate the model on the independent test set.
I also produce the test set output masks here, which are stored in output_plots.
Currently, I only plot the predicted mask for the final flight.

inferencing.py is where I set up the model to evaluate all examples in the inferencing directory specified in the config.
The directory I point to is a local one containing a single field from my test set.
Currently, I only plot the predicted mask for the final flight.
