<img src="static/EXTS_Logo.png" width="125px" align="right">

# AMLD20 - Anomaly Detection

This repository provides the resources for the talk and accompanying hands-on exercises on **Anomaly Detection** at the [EPFL Extension School Workshop - Machine Learning and Data Visualization](https://appliedmldays.org/workshops/epfl-extension-school-workshop-machine-learning-and-data-visualization) at the [Applied Machine Learning Days 2020](https://appliedmldays.org/).

**Slides** for the workshop are available [here](https://docs.google.com/presentation/d/1Jg9rO_3dXwKzJyDOr2ley8Is5oWKE6D_aJJlJrpw0mw/present?usp=sharing).




**Dataset**

The data is based on the [KDD-CUP 1999 challenge](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html) on network intrusion detection. A description of the original task can be found [here](http://kdd.ics.uci.edu/databases/kddcup99/task.html). The data provided for this workshop has been adapted from the [NSL-KDD version](https://www.kaggle.com/hassan06/nslkdd).

**Anomaly detection**

Anomaly detection can be treated as a supervised classification task. However this approach struggles when the portion of anomalies (here network attacks) is small. Instead we showcase an approach using [Isolation Forests](https://www.youtube.com/watch?v=RyFQXQf4w4w). 

The user can select the size of training dataset and vary its contamination rate, including a dataset without any anomalies. The model is then trained on this dataset and used to predict anomalies on a separate test set and evaluate the performance.

**Hands-on exercises**

The simplest way to run the hands-on exercises with **Google's Colab** or **Binder** in the cloud and interacting with them through your browser. Alternatively, you can choose to take a look at the already executed notebook in the **Offline View**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/epfl-exts/amld20-anomaly-detection/blob/master/AMLD20_anomalies_detection.ipynb) 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/epfl-exts/amld20-anomaly-detection/master?filepath=AMLD20_anomalies_detection.ipynb)
[![Generic badge](https://img.shields.io/badge/Offline_View-Open-Blue.svg)](https://nbviewer.jupyter.org/github/epfl-exts/amld20-anomaly-detection/blob/master/static/AMLD20_anomalies_detection_view.ipynb)

**Getting started:** 

If you are using **Colab** you need to execute the first cell. Otherwise you can skip this and start with loading settings and functions. If you want to execute a cell, make sure it is selected and then press `SHIFT`+`ENTER` or the `'Play'` button.