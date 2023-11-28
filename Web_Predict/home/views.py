from django.shortcuts import render, redirect
from django.http import HttpResponse
import os
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from TwoLayerNet import TwoLayerNet
from sklearn.preprocessing import StandardScaler


def mainpage(request):
    context = {}
    return render(request, 'home/index.html', context)

def credit(request):
    return render(request, 'home/credit.html')

def submit_data(request):
    if request.method == 'POST':
        # POST 요청에서 데이터 추출
        age = float(request.POST.get('Age'))
        sex = request.POST.get('Sex')
        chest_pain_type = request.POST.get('ChestPainType')
        cholesterol = float(request.POST.get('Cholesterol'))
        fasting_bs = float(request.POST.get('FastingBS'))
        max_hr = float(request.POST.get('MaxHR'))
        exercise_angina = request.POST.get('ExerciseAngina')
        oldpeak = float(request.POST.get('Oldpeak'))
        st_slope = request.POST.get('ST_Slope')

        # 추출한 데이터를 사용하여 예측 또는 다른 작업 수행
        data = [age, cholesterol, fasting_bs, max_hr, oldpeak, sex=='F', sex=='M', chest_pain_type=='ASY', chest_pain_type=='ATA', chest_pain_type=='NAP', chest_pain_type=='TA', exercise_angina=='N', exercise_angina=='Y', st_slope=='Down', st_slope=='Flat', st_slope=='Up']
        data = np.array(data, dtype=float)
        data = data.reshape(1, -1)

        current_directory = os.path.dirname(os.path.abspath(__file__))
        model_directory = os.path.join(current_directory, 'best_model.npz')
        scaler_directory = os.path.join(current_directory, 'scaler.pkl')

        with open(scaler_directory, 'rb') as f:
            loaded_scaler = pickle.load(f)

        numeric_data = data[:, :5]
        scaled_numeric_data = loaded_scaler.fit_transform(numeric_data)

        data[:, :5] = scaled_numeric_data

        network = TwoLayerNet(input_size=16, hidden_size1=20, hidden_size2=60, hidden_size3=30, output_size=1)

        network.load_model(model_directory)
        y = network.predict(data)
        y = network.sigmoid(y)
        y = network.binary_classification_predictions(y)
        result = y

        if result == 1:
            return HttpResponse("당신은 심부전일 가능성이 높습니다.")
        else:
            return HttpResponse("당신은 심부전일 가능성이 낮습니다.")
    else:
        return HttpResponse("Invalid request method.")