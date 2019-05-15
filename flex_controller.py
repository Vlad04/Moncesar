import os
from flask import Flask, render_template, flash, request, request, redirect, url_for
#from flask_wtf import FlaskForm
#from wtforms import StringField, SubmitField
#from classifier import Classifier
from werkzeug.utils import secure_filename
from flex import TrainingData

import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import operator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

UPLOAD_FOLDER = '/home/watsalacanoa/10semestre'
ALLOWED_EXTENSIONS = set(['csv'])
template = "prediction.html"

flex = TrainingData()
classifier = flex.training_data_loader()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    result_errors = []
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            result_errors = flex.do_classification(filename, classifier)
            if (result_errors):
                new_file_name = filename + "_error.txt"
                errfile = open(new_file_name, 'w')

                errfile.write("Predicted errors in  components:\n")
                for row in result_errors:
                    errfile.write(row + '\n')
                errfile.close()
    return render_template(template, elements=result_errors)
            #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #return redirect(url_for('uploaded_file',
            #                        filename=filename))
#    return '''
#    <!doctype html>
#    <title>Upload new File</title>
#    <h1>Upload new File</h1>
#    <form method=post enctype=multipart/form-data>
#      <input type=file name=file>
#      <input type=submit value=Upload>
#    </form>
#
#    '''
#
