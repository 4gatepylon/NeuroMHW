import os
# import pymongo
import json
import random
import hashlib
import time
import requests
import numpy as np

from flask import Flask, request, abort, Response

from flask_cors import CORS


from hashlib import sha256

app = Flask(__name__)
CORS(app)


def hashthis(st):


    hash_object = hashlib.md5(st.encode())
    h = str(hash_object.hexdigest())
    return h




@app.route('/simple', methods=['GET', 'POST'])
def simple():
    # js = transcribe()
    request_json = request.get_json()
    
    action = request['action']
    
    retjson = {}


    if action == "eegraw":
        doaction = "none"

        data = {}
        
        eeg = request_json['eegdata']
        
        amplitude = np.array(eeg)
        
        samplerate = request_json['samplerate']
        # duration = request_json['samplerate']
        
        fourierTransform = np.fft.fft(amplitude)/len(amplitude)           # Normalize amplitude

        fourierTransform = fourierTransform[range(int(len(amplitude)/2))] # Exclude sampling frequency

        

        tpCount     = len(amplitude)

        values      = np.arange(int(tpCount/2))

        timePeriod  = tpCount/samplerate

        frequencies = values/timePeriod
        
        alpha = 0
        beta = 0
        gamma = 0
        delta = 0
        theta = 0
        
        for x,y in np.c_[frequencies,abs(fourierTransform)]:
            if x<5:
                delta = delta + y
            if x>=5 and x <12:
                theta = theta + y
            if x>=12 and x <16:
                alpha = alpha + y
            if x>=16 and x <39:
                beta = beta + y
            if x>=39:
                gamma = gamma + y
        
        
        if alpha >= 25: ## CHANGE THIS VALUE TO THRESHOLD
            doaction = "motor"
        

        

        retjson['doaction'] = doaction
        retjson['data'] = data

        return json.dumps(retjson)



    if action == "doNothing":
        doaction = "none"

        data = {}
        
        ecg = []
        eeg = []
        emg = []

        for i in range(50):
            datai = random.randint(0, 1024)
            ecg.append(datai)
            datai = random.randint(0, 1024)
            eeg.append(datai)
            datai = random.randint(0, 1024)
            emg.append(datai)
        
        data['ecg'] = ecg
        data['eeg'] = eeg
        data['emg'] = emg
        

        retjson['doaction'] = doaction
        retjson['data'] = data

        return json.dumps(retjson)


    if action == "doBeep":
        doaction = "sound"

        data = {}
        
        ecg = []
        eeg = []
        emg = []

        for i in range(50):
            datai = random.randint(0, 1024)
            ecg.append(datai)
            datai = random.randint(0, 1024)
            eeg.append(datai)
            datai = random.randint(0, 1024)
            emg.append(datai)
        
        data['ecg'] = ecg
        data['eeg'] = eeg
        data['emg'] = emg
        

        retjson['doaction'] = doaction
        retjson['data'] = data
        return json.dumps(retjson)


    if action == "doNudge":
        doaction = "motor"

        data = {}
        
        ecg = []
        eeg = []
        emg = []

        for i in range(50):
            datai = random.randint(0, 1024)
            ecg.append(datai)
            datai = random.randint(0, 1024)
            eeg.append(datai)
            datai = random.randint(0, 1024)
            emg.append(datai)
        
        data['ecg'] = ecg
        data['eeg'] = eeg
        data['emg'] = emg
        

        retjson['doaction'] = doaction
        retjson['data'] = data

        return json.dumps(retjson)


    if action == "doPopup":
        doaction = "notify"

        data = {}
        
        ecg = []
        eeg = []
        emg = []

        for i in range(50):
            datai = random.randint(0, 1024)
            ecg.append(datai)
            datai = random.randint(0, 1024)
            eeg.append(datai)
            datai = random.randint(0, 1024)
            emg.append(datai)
        
        data['ecg'] = ecg
        data['eeg'] = eeg
        data['emg'] = emg
        

        retjson['doaction'] = doaction
        retjson['data'] = data

        return json.dumps(retjson)



    
    retjson['status'] = "command unknown"
    return json.dumps(retjson)



@app.route('/', methods=['GET', 'POST'])
def hello_world():
    # js = transcribe()
    
    js = {}
    js['status'] = 'done'


    resp = Response(js, status=200, mimetype='application/json')

    print ("****************************")
    print (resp)

    return js
    # return resp









@app.route("/dummyJson", methods=['GET', 'POST'])
def dummyJson():

    res = request.get_json()
    print (res)

    resraw = request.get_data()
    print (resraw)

##    args = request.args
##    form = request.form
##    values = request.values

##    print (args)
##    print (form)
##    print (values)

##    sres = request.form.to_dict()


    status = {}
    status["server"] = "up"
    status["request"] = res 

    statusjson = json.dumps(status)

    print(statusjson)

    js = "<html> <body>OK THIS WoRKS</body></html>"

    resp = Response(statusjson, status=200, mimetype='application/json')
    ##resp.headers['Link'] = 'http://google.com'

    return resp


if __name__ == '__main__':
    # app.run()
    # app.run(debug=True, host = '45.79.199.42', port = 8005)
    app.run(debug=True, host = 'localhost', port = 8005)  ##change hostname here
