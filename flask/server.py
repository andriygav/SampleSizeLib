# -*- coding: utf-8 -*- 
from flask import render_template, Flask, request

import tempfile
import re
import os

from time import gmtime, strftime

import pandas as pd
import numpy as np

from samplesizelib.utility import LinearSampleSizeEstimator
from samplesizelib.linear.models import LogisticModel, RegressionModel

app = Flask(__name__)

@app.route('/')
@app.route('/check')
def check():
    return render_template('check.html')

@app.route('/checker', methods = ['GET', 'POST'])
def checker():
    status = 'Dataset is not downloaded.'
    
    if request.method == 'POST':
        status = 'Dataset is loaded but not processed, please try again.'
        
        tables = None
        
        f = request.files['file']
        uploaded_filename = f.filename
        if '.' in uploaded_filename:
            extension = f.filename.rsplit('.')[-1]
            
            new_file, filename = tempfile.mkstemp('.{}'.format(extension))
            os.write(new_file, f.read())
            os.close(new_file)
            
            if extension == 'csv':
                status = 'Dataset is received but not processed, please try again.'

                dataset = pd.read_csv(filename)
                if 'y' not in dataset:
                    status = 'Target column "y" is not specified'
                else:
                    y = dataset['y'].to_numpy()
                    del dataset['y']
                    X = dataset.to_numpy()

                estimator = LinearSampleSizeEstimator(LogisticModel)

                ret = estimator(X, y)

                tabledict = dict()
                for key in ret:
                    for name in ret[key]:
                        tabledict[name] = {'m*': ret[key][name]['m*']}

                dataframe = pd.DataFrame(tabledict)
                tables = [dataframe.to_html(classes='data')]

                status = None

            else:
                status = 'Invalid file extension: "{}".'.format(extension)
                
            os.remove(filename)
           
        else:
            status = 'Invalid file extension: "{}".'.format('')
        
        
    return render_template('checker.html', status=status,  tables=tables)
        
        
    