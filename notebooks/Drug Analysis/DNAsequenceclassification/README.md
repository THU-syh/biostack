### Summary: 

- This project aims to build neural network models to classify the resistance status of genes against a certain drug. The given dataset contains 100,000 gene sequences and their resistance (T/F). 

### Data preparation: 

- The DNA bases (characters) are converted to integers.
- The genes were padded to have equal lengths. 
- The resistance is one-hot encoded.

### Models in [CNNandRNNwithoutCV.ipynb](https://github.com/euniceky/DNAsequenceclassification/blob/master/CNNandRNNwithoutCV.ipynb): 

- In a separate notebook, we built <b>model1 </b> which was a CNN with dropout layers. This model had an underfitting issue.

- In a separate notebook, we built <b> model2 </b> which was a CNN without dropouts. Although the hyperparameters were randomly chosen, this model produced the <b> best result (99.46% accuracy)</b>.

- In a separate notebook, we built <b> model3 </b> which was an RNN. We did not have a GPU, and we downsampled and worked with 20,000 data points.
- This initial build did not perform as well compared to CNN models. 
- We did not pursue this route further as the computation cost for RNNs is high. 

### Model in [CNN_CV.ipynb](https://github.com/euniceky/DNAsequenceclassification/blob/master/CNN_CV.ipynb) (this is the main file): 

- This notebook contains a CNN <b> model </b> which has two filter layers, two pooling layers, and one hidden layer. 
- We performed 5-fold cross-validation on the training set to find the best hyperparameters. 
- The hyperparameters are: the number of filters, the filter sizes (kernel sizes), the pooling size, the number of neurons in the hidden layer (units), and the optimizer. 
- The best 5-fold cv validation score is 95.28% 
- With the best parameters, the <b> model </b> predicts the resistance with <b> 97.13% accuracy </b>.




