###################
Service Flask Usage
###################

Requirements
============

- Python 3.6.2
- pip 20.0.2

Install package requirements
----------------------------

.. code-block:: bash

    python3.6 -m pip install -r flask/requirements.txt

Run Service
===========
Write next commant to the terminal:

.. code-block:: bash

    bash flask.sh 8080

where 8080 is a port where service runnning. After running, follow the link `link <http://localhost:8080>`_ page.

Service Usage
=============
To analyze the sample size for given data, upload a csv file with data in the following format:

.. code-block:: bash

	y,x_1,x_2
	1.0,1.76,0.4
	0.0,0.98,2.24
	1.0,1.87,-0.98
	0.0,0.95,-0.15
	1.0,-0.1,0.41
	0.0,0.14,1.45
	0.0,0.76,0.12
	0.0,0.44,0.33
	0.0,1.49,-0.21
	0.0,0.31,-0.85
	1.0,-2.55,0.65

The CSV file must contain the target variable in the column named 'y'. Feature columns can have any names.
