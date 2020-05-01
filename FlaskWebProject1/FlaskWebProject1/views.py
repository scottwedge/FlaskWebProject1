"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template
from FlaskWebProject1 import app
from FlaskWebProject1.datapresent import *
from flask import request, redirect
from flaskext.markdown import Markdown
import pandas as pd
import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn  as sns

banks = bankmarketing()

@app.route('/')
def home():

    Markdown(app)

    return render_template(
        'index.html',
         title='Home Page',
         year=datetime.now().year,
         tables=[banks.dataframe().head().to_html(classes='table table-bordered')],
         )

@app.route('/analysis')
def analysis():
    data = banks.bank.iloc[: , 0:7]
    age_min = data['age'].min()
    age_max = data['age'].max()

    quart1 = data['age'].quantile(q = 0.25)
    quart2 = data['age'].quantile(q = 0.50)
    quart3 = data['age'].quantile(q = 0.75)
    quart4 = data['age'].quantile(q = 1.00)
    outliners = data['age'].quantile(q = 0.75) + 1.5*(data['age'].quantile(q = 0.75) - data['age'].quantile(q = 0.25))
    age_mean = round(data['age'].mean())
    age_std = round(data['age'].std())

    Markdown(app)
    return render_template(
        'analysis.html',
         year=datetime.now().year,
         tables=[data.head().to_html(classes='table table-bordered')],
         jobs=data['job'].unique(),
         marital=data['marital'].unique(),
         education=data['education'].unique(),
         default=data['default'].unique(),
         housing=data['housing'].unique(),
         loan=data['loan'].unique(),
         age_min=age_min,
         age_max=age_max,
         age_mean=age_mean,
         age_std=age_std,
         quart1=quart1,
         quart2=quart2,
         quart3=quart3,
         quart4=quart4
         )

@app.route('/linear')
def linear():
    accuracy = {'logres': banks.logregression()}
    Markdown(app)

    return render_template(
        'linear.html',
         year=datetime.now().year,
         )

@app.route('/xgboost')
def xgboost():

    Markdown(app)

    return render_template(
        'xgboost.html',
         year=datetime.now().year,
         )

@app.route('/kmeans')
def kmeans():

    Markdown(app)

    return render_template(
        'kmeans.html',
         year=datetime.now().year,
         )

@app.route('/knn')
def knn():

    Markdown(app)

    return render_template(
        'knn.html',
         year=datetime.now().year,
         )


@app.route('/plot_age1.png')
def plot_age1():
    fig = create_figure1()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure1():
    data = banks.bank.iloc[: , 0:7]
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 8)
    sns.countplot(x = 'age', data = data)
    ax.set_xlabel('Age', fontsize=15)
    ax.set_ylabel('Count', fontsize=15)
    ax.set_title('Age Count Distribution', fontsize=15)
    sns.despine()
    return fig

@app.route('/plot_age2.png')
def plot_age2():
    fig = create_figure2()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure2():
    data = banks.bank.iloc[: , 0:7]
    fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 5))
    sns.boxplot(x = 'age', data = data, orient = 'v', ax = ax1)
    ax1.set_xlabel('People Age', fontsize=15)
    ax1.set_ylabel('Age', fontsize=15)
    ax1.set_title('Age Distribution', fontsize=15)
    ax1.tick_params(labelsize=15)

    sns.distplot(data['age'], ax = ax2)
    sns.despine(ax = ax2)
    ax2.set_xlabel('Age', fontsize=15)
    ax2.set_ylabel('Occurence', fontsize=15)
    ax2.set_title('Age x Ocucurence', fontsize=15)
    ax2.tick_params(labelsize=15)

    plt.subplots_adjust(wspace=0.5)
    plt.tight_layout() 
    return fig