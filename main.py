from flask import Flask, render_template,request,redirect
from werkzeug.utils import secure_filename
import base64
import cv2
import numpy as np
from detect import numplatedetect
from recongition import show_results
from segmentation import segment_characters


data_list = []
alpha = []

app = Flask("__init__",static_folder='static')


@app.route('/')
def index():
    return render_template('index.html')



@app.route('/login',methods=['GET','POST'])
def login():
    if request.method == "POST":
        username = request.form["username"]
        return redirect(f'/hp/{username}')
    return render_template('login.html')

# @app.route('/ma')
# def mapqw():
#     alpha = []
#     for i in range(35):
#         alpha[i] = 0
#     return render_template('map.html',alpha=alpha)

@app.route('/lop',methods=['GET','POST'])
def lj():
    return render_template('map.html',alpha=alpha,data = data_list)


@app.route('/l')
def jio():
    return "nn"

@app.route('/hp/<username>',methods=['GET','POST'])
def home(username):
    if request.method == "POST":
        f = request.form['qwerty']
        name = request.form['pname']
        phone = request.form['phone']
        slot = request.form['slot']

        

        alpha.append(int(slot))

        decoded_data = base64.b64decode(f)
        np_data = np.fromstring(decoded_data,np.uint8)
        img = cv2.imdecode(np_data,cv2.COLOR_BGR2RGB)
        op = numplatedetect(img)
        sc = segment_characters(op)
        result = show_results(sc)
        string = base64.b64encode(cv2.imencode('.jpg', op)[1]).decode()
   
        data_list.append({
            "f":string,
            "name":name,
            "phone":phone,
            "slot":slot,
            "result":result
        })
        print(result)
        return redirect("/lop")

    return render_template("home.html",name=username)


if __name__ == "__main__":
    app.run(host="0.0.0.0")