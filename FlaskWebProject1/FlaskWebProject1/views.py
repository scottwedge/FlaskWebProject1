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
    data = banks.bank.iloc[: , 0:-1]
    table = banks.bank.iloc[: , 0:7]
    table_related = banks.bank.iloc[: , 7:11]
    other_attributes = banks.bank.loc[: , ['emp.var.rate', 'cons.price.idx',
                                          'cons.conf.idx', 'euribor3m', 'nr.employed',
                                          'campaign', 'pdays','previous', 'poutcome']]

    age = list_creator(data['age'].min(), data['age'].max(), round(data['age'].std()), round(data['age'].mean()))

    outliners = data['age'].quantile(q = 0.75) + 1.5*(data['age'].quantile(q = 0.75) - data['age'].quantile(q = 0.25))
    quart = list_creator(data['age'].quantile(q = 0.25), data['age'].quantile(q = 0.50),
                        data['age'].quantile(q = 0.75), data['age'].quantile(q = 1.00), outliners)

    unique = list_creator(data['job'].unique(), data['marital'].unique(), data['education'].unique(),
                          data['default'].unique(), data['housing'].unique(), data['loan'].unique())

    unique_related = list_creator(data['contact'].unique(), data['month'].unique(),
                                 data['day_of_week'].unique()) 

    duration_call = list_creator(round(data['duration'].min()/60), round(data['duration'].max()/60),
                                 round(data['duration'].std()/60), round(data['duration'].mean()/60))
    quart_duration = list_creator(data['duration'].quantile(q = 0.25), data['duration'].quantile(q = 0.50),
                                  data['duration'].quantile(q = 0.75), data['duration'].quantile(q = 1.00))

    return render_template(
        'analysis.html',
         year=datetime.now().year,
         tables=[table.head().to_html(classes='table table-bordered')],
         table_related=[table_related.head().to_html(classes='table table-bordered')],
         unique=unique,
         age=age,
         quart=quart,
         unique_related=unique_related,
         duration_call=duration_call,
         quart_duration=quart_duration,
         others=[other_attributes.head().to_html(classes='table table-bordered')],
         )

@app.route('/linear')
def linear():
    r2, mae, mse, coef, linpred = banks.linear()

    return render_template(
        'linear.html',
         year=datetime.now().year,
         r2=r2,
         mae=mae,
         mse=mse,
         coef=coef
         )

@app.route('/xgboost')
def xgboost():
    conf_xg, accuracy_xg = banks.xgboost()
    accuracy_xg = (float(int(accuracy_xg*1000)))/1000

    return render_template(
        'xgboost.html',
         year=datetime.now().year,
         conf_xg=conf_xg,
         accuracy_xg=accuracy_xg
         )

@app.route('/kmeans')
def kmeans():
    centers = banks.cluster_centers
    
    return render_template(
        'kmeans.html',
         year=datetime.now().year,
         centers=centers,
         )

@app.route('/knn')
def knn():
    conf, accuracy = banks.knn()
    accuracy = (float(int(accuracy*1000)))/1000
    return render_template(
        'knn.html',
         year=datetime.now().year,
         conf=conf,
         accuracy=accuracy
         )


@app.route('/plot_age1.png')
def plot_age1():
    fig = banks.plot_age_1()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/plot_age2.png')
def plot_age2():
    fig = banks.plot_age_2()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/plot_jobs.png')
def plot_jobs():
    fig = banks.plot_jobs()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/plot_marital.png')
def plot_marital():
    fig = banks.plot_marital()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/plot_education.png')
def plot_education():
    fig = banks.plot_education()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/plot_DHL.png')
def plot_DHL():
    fig = banks.plot_DHL()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/plot_duration.png')
def plot_duration():
    fig = banks.plot_duration()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/plot_con_day_month.png')
def plot_con_day_month():
    fig = banks.plot_con_day_month()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


@app.route('/plot_kmeans_1.png')
def plot_kmeans_1():
    fig = banks.plot_kmeans_1()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/plot_kmeans_2.png')
def plot_kmeans_2():
    fig = banks.plot_kmeans_2()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/plot_knn_conf.png')
def plot_knn_conf():
    fig = banks.plot_knn_conf()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/plot_xgboost_conf.png')
def plot_xgboost_conf():
    fig = banks.plot_xgboost_conf()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/plot_linear.png')
def plot_linear():
    fig = banks.plot_linear()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def list_creator(*args):
    return list(args)