# -*- coding: utf-8 -*- 
from flask import render_template, Flask, request, jsonify, Response

from threading import Thread

import tempfile
import time
import json
import re
import os

from time import gmtime, strftime

import pandas as pd
import numpy as np

from api import worker, scheduler, get_config, NAME_TO_MODEL, NAME_TO_STATMODEL


app = Flask(__name__)

shed = scheduler()

@app.route('/')
@app.route('/check')
def check():
    config = get_config()
    return render_template('check.html', config=config)

@app.route('/get_result/<int:id>')
def get_result(id):
    model = shed.get_job(id)
    if model is None:
        response = jsonify({})
        response.status_code = 400

    elif model.result is None:
        response = jsonify({})
        response.status_code = 400
    else:
        tabledict = dict()
        for key in model.result:
            tabledict[key] = {'m*': model.result[key]['m*']}

        dataframe = pd.DataFrame(tabledict)
        table = dataframe.to_html(classes='data')

        response = jsonify(table=table)
        response.status_code = 200
    return response

@app.route('/result/<int:id>')
def result(id):
    model = shed.get_job(id)
    if model is None:
        status = 'Id not recognise.'
        tables = None
    elif model.result is None:
        status = 'Result not ready, please wait.'
        tables = None
    elif model.status is not None:
        status = model.status
        tables = None
    else:
        tabledict = dict()
        for key in model.result:
            tabledict[key] = {'m*': model.result[key]['m*']}

        dataframe = pd.DataFrame(tabledict).T
        tables = [dataframe.to_html(classes='data')]
        status = None

    return render_template('result.html', status=status, tables=tables, _id=id)

@app.route('/progress/<int:id>')
def progress(id):
    def generate():
        response = dict()
        response['persentage'] = 0.
        response['progress'] = dict()
        status = None
        while response['persentage'] < 100 and status is None:
            model = shed.get_job(id)
            if model is None:
                return 'data:{}\n\n'.format(json.dumps(response))
            elif model.progress is None:
                return 'data:{}\n\n'.format(json.dumps(response))
            else:
                status = model.status
                response['persentage'] = model.percentage()
                for key in model.progress:
                    if key not in response['progress']:
                        response['progress'][key] = dict()
                    response['progress'][key]['progress'] = 0.
                    response['progress'][key]['status'] = 'none'
                    response['progress'][key]['m*'] = 'none'
                    
                    response['progress'][key]['status'] = model.progress[key]['status']
                    response['progress'][key]['progress'] = int(model.models[key].status())

                    if 'result' in model.progress[key]:
                        response['progress'][key]['m*'] = int(model.progress[key]['result']['m*'])

            text = 'data:{}\n\n'.format(json.dumps(response))
            yield text
            time.sleep(1)

    return Response(generate(), mimetype= 'text/event-stream')

@app.route('/checker', methods = ['GET', 'POST'])
def checker():
    status = 'Dataset is not downloaded.'
    _id = None
    tabledict = dict()
    info = []
    
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

            job = worker(statmodel, config, X, y)
            if job.status is not None:
                return render_template('checker.html', status=job.status)
            _id = shed.add_job(job)

            thread = Thread(target=job.forward)
            thread.daemon = True
            thread.start()

            for key in job.progress:
                tabledict[key] = dict()
                info = ['progress', 'status', 'm*']
                for item in info:
                    tabledict[key][item] = 'none'
                tabledict[key]['progress'] = 0

            status = None
        else:
            status = 'Invalid file extension: "{}".'.format(extension)
            
        os.remove(filename)

    return render_template('checker.html', status=status, tabledict=tabledict, info=info, _id=_id)
            
        
        
    
        
        
    