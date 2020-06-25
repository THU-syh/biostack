# DL4GP

<b>DL4GP</b> (Deep Learning for Genomics and Proteonomics) is a project which leverages Deep Learning to learn patterns from genetic and protein sequence. We propose neural network architectures which can perform variety of tasks using NLP like techniques such as identifying sequences and drug discovery. 

The data folder contains the dataset of seuences and associated data. It was obtained from the [Kaggle structural protein dataset](https://www.kaggle.com/shahir/protein-data-set).
The original source of the data is the [RCSB PDB](http://www.rcsb.org/pdb/).

The [iPython notebook](https://github.com/ritabratamaiti/DL4GP/blob/master/Visualization%20Notebook.ipynb) will let you visualize the data. However, since I am uploading the notebook directly from kaggle the required code to access the data will vary for you. You will have to donload the zip files from the data folder, unzip them and then use pandas to read the csv files.
It is pretty easy to see which dataframe is associated with which file. Once the dataframes are loaded, you can run each cell and view protein data analysis.
If you don't want to go into all that effort (xD), I have made a [public kaggle kernel](https://www.kaggle.com/ritabratamaiti/visualization-script-and-simple-models?scriptVersionId=10332509) which is basically the same notebook along with the data loaded into the environment.
You can modify the code and play around the data using Kaggle's Nvidia K80 GPUs =)

# Important Note:

I am still working on the neural network which will be used to learn and classify protein sequences. Once this is done, I will upload the python notebook containing the code
used to make the model, as well as a trained model and embedding weights.
