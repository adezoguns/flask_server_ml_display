from flask import Flask, url_for, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.utils import secure_filename
from werkzeug.serving import run_simple
from id_class_locator import id_class_detector
import os
import time
from cv2 import cv2

app=Flask(__name__)
#app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///my.db'
#db=SQLAlchemy(app)
path2File= os.path.dirname(os.path.realpath(__file__))
pathToModel=path2File+'/WorkArea/FRCNN'

PATH = path2File+'/static/input'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
app.config['PATH']=PATH
#app.config["TEMPLATES_AUTO_RELOAD"] = True

model = cv2.dnn.readNetFromTensorflow(pathToModel+'/frozen_inference_graph.pb', pathToModel+'/frcnn.pbtxt')

@app.route('/hello', methods=['POST', 'GET'])
def hello():
	return('Hello')

@app.route('/', methods=['POST', 'GET'])
def index():
	return render_template('home.html')


@app.route('/upload', methods=['POST', 'GET'])
def upload():
	if request.method == 'POST':
		# check if the post request has the file part
		file = request.files['imageUploadForm']     
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['PATH'], filename))
		print(filename)
		img=cv2.imread(os.path.join(app.config['PATH'], filename))
		id_class_detector(img, model, filename, debug=False)
		#time.sleep(2)
		
	return render_template('home.html', value=filename)
	
if __name__=="__main__":
	run_simple('127.0.0.1', 9100, app, use_reloader=False)
