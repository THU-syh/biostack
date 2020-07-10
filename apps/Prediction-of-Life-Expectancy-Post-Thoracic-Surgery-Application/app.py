import numpy as np
from flask import Flask, request, jsonify, render_template,session
import pickle
from flask_mysqldb import MySQL
import yaml
import re


app = Flask(__name__)

app.secret_key = 'tapas'

#Configure db
db = yaml.load(open('config.yaml'))
app.config['MYSQL_HOST'] = db['mysql_host']
app.config['MYSQL_USER'] = db['mysql_user']
app.config['MYSQL_PASSWORD'] = db['mysql_password']
app.config['MYSQL_DB'] = db['mysql_db']

mysql = MySQL(app)




model = pickle.load(open('model.pki','rb'))
model.n_jobs = 1
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.php')

@app.route('/register', methods = ['GET','POST'])
def register():
    msg = ''
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        confpass = request.form['confpass']

        cursor = mysql.connection.cursor()
        cursor.execute('SELECT * FROM user WHERE name = %s', (username,))
        users = cursor.fetchone()

        if users:
            msg = 'Account already exists'

        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid Email Id'

        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'

        elif not username or not password or not email:
            msg = 'Please fill out the form'

        elif password != confpass:
            msg = 'The passwords do not match'

        else:
            # The account does not exist and hence we can add it to the database
            cursor.execute('INSERT INTO user VALUES (%s, %s, %s)', (username, email, password,))
            mysql.connection.commit()
            msg = 'You Have Been Successfully Registered'





    elif request.method == 'POST':
        msg = 'Please fill out the form'
    
    return render_template('index.html',msg= msg)



    

@app.route('/welcome', methods = ['GET','POST'])
def welcome():
    msg = ''
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']
        cursor = mysql.connection.cursor()
        cursor.execute('SELECT * FROM user WHERE email = %s AND password = %s', (email, password,))
        users = cursor.fetchone()
        if users:
            session['loggedin'] = True
            #session['name'] = users['name']
            return render_template('welcome.php')

        else:
            msg = 'Incorrect Usrname/password'
            return msg
    else:
        return 'Post not found'


    

@app.route('/predict', methods = ['POST'])
def predict():
    
    final_features = [0,0,0,0,0]
    final_features[0] = (int(request.form.get("Performance")))
    final_features[1] =(int(request.form.get("Dyspnoea")))
    final_features[2] = (int(request.form.get("Cough")))
    final_features[3]= (int(request.form.get("Tumour_Size")))
    final_features[4] = (int(request.form.get("Diabetes_Mellitus")))
    name = request.form.get("fname").strip(" ")
    features = (np.array(final_features))
    features = features.reshape(1,-1)
    prediction = model.predict(features)
    
    output = prediction[0]
    
    if output == 0:
        return render_template('prediction.php',predicted_text = 'As per our predictions, {}, you can go ahead with the surgery'.format(name))
    else:
        return render_template('prediction.php',predicted_text = 'As per our predictions, {}, you are advised to not go ahead with the surgery'.format(name))
    

@app.route('/logout')
def logout():
    return render_template('index.html')



if __name__ == "__main__":
    app.run(debug=True)