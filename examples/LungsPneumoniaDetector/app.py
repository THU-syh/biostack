from flask import Flask, render_template,request
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

dict = {0:'Normal',1:'Pneumoniua'}
app =  Flask(__name__)

model = load_model('static/model.h5')
model._make_predict_function()
@app.route('/')
def landingPage():
	return render_template("index.html")

@app.route('/detect')
def detect():

	#if request.method == 'POST':
	#	f = request.files("userfile")
	#	path = "./static/{}0".format(f.filename)
	#	f.save(path)
		

	return render_template("detect.html")

@app.route('/pred',methods=['POST'])
def pred():
	path=""
	if request.method == 'POST':
		f = request.files["userfile"]
		path = "./static/{}".format(f.filename)
		f.save(path)
		img = image.load_img(path,target_size=(224,224,3))
		test = image.img_to_array(img)
		x = np.expand_dims(test, axis=0)
		r = model.predict(x)
		model._make_predict_function()
		result_dict = {
		'image': path,
		'detect': dict[np.argmax(r)]
		}
		
	return render_template("detect.html",result=result_dict)
		
if __name__ == '__main__':
	app.run(debug=True)