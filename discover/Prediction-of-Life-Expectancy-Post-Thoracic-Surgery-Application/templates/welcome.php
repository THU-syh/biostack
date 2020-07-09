<?php 
  session_start(); 

  if (!isset($_SESSION['email'])) {
  	$_SESSION['msg'] = "You must log in first";
  	header('Location: login.php');
  }
  if (isset($_GET['logout'])) {
  	session_destroy();
  	unset($_SESSION['username']);
  	unset($_SESSION['email']);
  	unset($_SESSION['success']);
  	unset($_GET['logout']);
  	header('Location: login.php');
  }
?>

<!DOCTYPE html>
<html>
<head>
	<title>Home</title>
	<link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
	<script src="https://kit.fontawesome.com/a076d05399.js"></script>
  <link href="{{ url_for('static', filename='bootstrap/css/bootstrap.min.css') }}" rel="stylesheet" id="bootstrap-css">

  <link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='trial.css')}}">
  <link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='homepage.css')}}">
  <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&family=Righteous&display=swap" rel="stylesheet">

</head>
<body>

<div class="header w3-card w3-margin w3-padding-large">
	<!-- Logout -->
    	<p><a href="{{ url_for('logout') }}" class="logout" style="text-decoration: none; float: right;">logout <i class="fas fa-sign-out-alt"></i></a> </p>
	<h2 class="title">POST-OPERATIVE LIFE EXPECTANCY FOR LUNG CANCER PATIENTS</h2>

<div>
    <!-- logged in user information -->
    
</div>
</div>
<center>
    <div class="assessment-container container">
        <div class="row">
            <div class="col-md-6 form-box">
                <form role="form" class="registration-form" action="{{url_for('predict')}}" method="POST">
                  <!-- Patient Details-->  
                    <fieldset>
                        <div class="form-top">
                            <div class="form-top-left">
                                <h3><span><i class="fa fa-calendar-check-o" aria-hidden="true"></i></span>Please provide patient details</h3>
                                <p>We aim to provide complete confidentiality.
                                </p>
                            </div>
                        </div>
                        <div class="form-bottom">
                            <div class="row">
                                <div class="form-group col-md-6 col-sm-6">
                                    <input type="text" class="form-control" name="fname" placeholder="Firstname" id="fname">
                                </div>
                                <div class="form-group col-md-6 col-sm-6">
                                    <input type="text" class="form-control" placeholder="Lastname" id="lname">
                                </div>
                            </div>
                            <div class="form-group" style="margin-bottom:3px;">
                                <div class="row">
                                    <div class="form-group col-md-9 col-sm-9">
                                        <input type="text" class="form-control" placeholder="Contact Number" id="contact_number">
                                    </div>
                                </div>
                            </div>
                            <div class="form-group">
                                <input type="text" name="email" placeholder="Email" class="form-email form-control" id="email" required>
                            </div>

                            <div class="form-group">
                                <select class="form-control">
                                    <option>Gender</option>
                                    <option>Male</option>
                                    <option>Female</option>
                                </select>
                            </div>
                            
                            <button type="button" class="btn btn-next">Next</button>
                        </div>
                    </fieldset>
                    <!-- Performance-->  
                    <fieldset>
                        <div class="form-top">
                            <div class="form-top-left">
                                <h3><span><i class="fa fa-calendar-check-o" aria-hidden="true"></i></span>Performance status on Zubrod scale</h3>
                                <p>Good (0) to Poor (2)
                                </p>
                            </div>
                        </div>
                        <div class="form-bottom">
                            <div class="row">
                                <div class="form-group col-md-6 col-sm-6">
                                    
                                    <select id="cars" name="Performance" class = 'form-control', id='performance'>
                                      <option value=0>0</option>
                                      <option value=1>1</option>
                                      <option value=2>2</option>
                                    </select>
                                </div>
                            </div>
                            <button type="button" class="btn btn-previous">Previous</button>
                            <button type="button" class="btn btn-next">Next</button>
                        </div>
                    </fieldset>
                    <!-- Dysnopea-->  
                    <fieldset>
                        <div class="form-top">
                            <div class="form-top-left">
                                <h3><span><i class="fa fa-calendar-check-o" aria-hidden="true"></i></span>Difficulty or labored breathing, before surgery</h3>
                                <p>If True, enter 1, else 0
                                </p>
                            </div>
                        </div>
                        <div class="form-bottom">
                            <div class="row">
                                <div class="form-group col-md-6 col-sm-6">
                                    <select id="dyspnoea" name="Dyspnoea" class = 'form-control'>
                                      <option value=0>False</option>
                                      <option value=1>True</option>
                                    </select>
                                </div>
                            </div>
                            <button type="button" class="btn btn-previous">Previous</button>
                            <button type="button" class="btn btn-next">Next</button>
                        </div>
                    </fieldset>
                     <!-- Cough-->  
                    <fieldset>
                        <div class="form-top">
                            <div class="form-top-left">
                                <h3><span><i class="fa fa-calendar-check-o" aria-hidden="true"></i></span>Symptoms of Coughing, before surgery</h3>
                                <p>If True, enter 1, else 0
                                </p>
                            </div>
                        </div>
                        <div class="form-bottom">
                            <div class="row">
                                <div class="form-group col-md-6 col-sm-6">
                                    <select id="cough" name="Cough" class = 'form-control'>
                                      <option value=0>False</option>
                                      <option value=1>True</option>
                                    </select>
                                </div>
                            </div>
                            <button type="button" class="btn btn-previous">Previous</button>
                            <button type="button" class="btn btn-next">Next</button>
                            
                        </div>
                    </fieldset>
                     <!-- Tumour-->  
                    <fieldset>
                        <div class="form-top">
                            <div class="form-top-left">
                                <h3><span><i class="fa fa-calendar-check-o" aria-hidden="true"></i></span>T in clinical TNM - size of the original tumor</h3>
                                <p>1 (smallest) to 4 (largest)
                                </p>
                            </div>
                        </div>
                        <div class="form-bottom">
                            <div class="row">
                                <div class="form-group col-md-6 col-sm-6">
                                    <select id="tumour" name="Tumour_Size" class = 'form-control'>
                                      <option value=1>1</option>
                                      <option value=2>2</option>
                                      <option value=3>3</option>
                                      <option value=4>4</option>
                                    </select>
                                </div>
                            </div>
                            <button type="button" class="btn btn-previous">Previous</button>
                            <button type="button" class="btn btn-next">Next</button>
                        </div>
                    </fieldset>
                     <!-- Diabetes-->  
                    <fieldset>
                        <div class="form-top">
                            <div class="form-top-left">
                                <h3><span><i class="fa fa-calendar-check-o" aria-hidden="true"></i></span>Type 2 diabetes mellitus</h3>
                                <p>If True, enter 1, else 0
                                </p>
                            </div>
                        </div>
                        <div class="form-bottom">
                            <div class="row">
                                <div class="form-group col-md-6 col-sm-6">
                                    <select id="diabetes" name="Diabetes_Mellitus" class = 'form-control'>
                                      <option value=0>False</option>
                                      <option value=1>True</option>
                                    </select>

                                </div>
                            </div>
                            <button type="button" class="btn btn-previous">Previous</button>
                            <button type="submit" class="btn"  >Submit to Make Prediction</button>
                        </div>
                    </fieldset>
                </form>
            </div>
        </div>
    </div>
</div>
</center>
<script src="{{ url_for('static', filename='bootstrap/js/bootstrap.min.js') }}"></script>

<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
<script type="text/javascript" src="{{ url_for('static', filename='trial.js') }}"></script>
</body>
</html>