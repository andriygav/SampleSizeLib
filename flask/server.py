# -*- coding: utf-8 -*- 
from flask import render_template, Flask, request, jsonify

import tempfile
import re
import os

from time import gmtime, strftime

import pandas as pd
import numpy as np

from api import get_config, NAME_TO_MODEL, NAME_TO_STATMODEL


app = Flask(__name__)

@app.route('/')
@app.route('/check')
def check():
    config = get_config()
    return render_template('check.html', config=config)

@app.route('/checker', methods = ['GET', 'POST'])
def checker():
    status = 'Dataset is not downloaded.'
    tables = None
    
    if request.method == 'POST':
        statmodel = NAME_TO_STATMODEL.get(request.form.get('statmodel', None), None)
        if statmodel is None:
            status = 'Linear model is not recognise: {}'.format(statmodel)
            return render_template('checker.html', status=status)

        default_config = get_config()
        config = dict()
        for key in default_config:
            if request.form.get('config[{}][use]'.format(key), 'off') == 'on':
                config[key] = dict()
                for param in default_config[key]:
                    default_config[key][param] = request.form.get('config[{}][{}]'.format(key, param), None)
                    if default_config[key][param]:
                        config[key][param] = float(default_config[key][param])

        print(config)

        status = 'Dataset is loaded but not processed, please try again.'

        f = request.files['file']
        uploaded_filename = f.filename
        if '.' not in uploaded_filename:
            status = 'Cant recognise file extension: "{}".'.format(uploaded_filename)
            return render_template('checker.html', status=status)

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

            models = dict()
            for key in config:
                try:
                    models[key] = NAME_TO_MODEL[key](statmodel, **config[key])
                except ValueError as e:
                    status = 'Model "{}" initialise error: {}'.format(key, str(e))
                    return render_template('checker.html', status=status)


            result = dict()

            for key in models:
                result[key] = models[key](X, y)

            tabledict = dict()
            for key in result:
                tabledict[key] = {'m*': result[key]['m*']}

            dataframe = pd.DataFrame(tabledict)
            tables = [dataframe.to_html(classes='data')]

            status = None
        else:
            status = 'Invalid file extension: "{}".'.format(extension)
            
        os.remove(filename)

    return render_template('checker.html', status=status,  tables=tables)
            
        
        
    
        
        
    