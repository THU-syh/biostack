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
  
</div>
</div>

	<h3 class="predict">{{ predicted_text }}</h3>
</div>
<script src="{{ url_for('static', filename='bootstrap/js/bootstrap.min.js') }}"></script>

<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
<script type="text/javascript" src="{{ url_for('static', filename='trial.js') }}"></script>
</body>
</html>