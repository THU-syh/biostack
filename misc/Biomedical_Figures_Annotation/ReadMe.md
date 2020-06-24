# Biomedical Figures Annotator 

A GUI based semi annotation tool to annotate chart figures according to their type (e.g. Bar chart, Line chart) from Biomedical literature (e.g. Pubmed).  

## Dependencies
* pyQt5
* numpy
* pandas
* scipy
* scikit-learn
* keras
* tensorflow

Run `pip install -r requirements.txt` to automatically install the dependencies in your virtual environment. Execute the `viewer.py` file. Tested with `python 3.6` and `tensorflow 1.4`. 

## Guide
* Select open file or open csv if you have already created from the toolbar.
* Select the folder which contains the papers along with the figure charts from pubmed.
* Annotate some images by pressing right click on the scenes. 
* Click the training button to train on the small subset in order to get better results.
* Continue the annotation.

### Example

<img src="https://github.com/dimimal/Biomedical_Figures_Annotation/blob/master/Pictures/show_4.png" width="512" height="512">

