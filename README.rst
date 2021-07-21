=================================================================================
NAT-ML: ML prediction of response to neoadjuvant breast cancer therapy
=================================================================================

.. sectnum::

.. contents:: Table of contents

Interactive setup
~~~~~~~~~~~~~~~~~~~~~~~~~

We have designed an interactive dashboard to apply the fully-integrated NAT response model on new (or any desired) data. The prediction is updated in real time when you modify any of the features.

To launch the tool and use it interactively, please click here (and be patient, it may take a little while!):

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/micrisor/NAT-ML.git/main?urlpath=%2Fvoila%2Frender%2Fvalidation_online%2Finteraction_prediction.ipynb


Reproducing the results
~~~~~~~~~~~~~~~~~~~~~~~~~~

This repository also contains all the code necessary to reproduce the results in the paper. In particular: 

* **training_code** contains the code necessary to train the models from scratch.
  
  * Run *schedule.py* to train all the standard models on an HPC cluster. 
  * Run *schedule_manualimportance.py* to train all the leave-one-out models on an HPC cluster.

- **trained_models** contains the trained model files so you can apply them directly on the validation data or any other dataset. 

  - Note that leave-one-out models (only necessary for feature importance figures) are too large to be stored on GitHub so you will need to re-create them locally as explained above.

* **validation_code** contains the code necessary to apply the trained models on the validation dataset. 
  
  * Run *execute.py* to apply the models.

- **graphs** contains Jupyter Notebooks that generate the panels in Figure 4 and Extended Figure 9.

  - If you want to generate the feature importance panels, you will need to generate the leave-one-out models first as explained above.
