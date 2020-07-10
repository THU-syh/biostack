<?php include('server.php');?>
<!DOCTYPE HTML>
<html>
<head>
	<title>Login</title>
	<link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
	<script src="https://cdn.jsdelivr.net/jquery/1.12.4/jquery.min.js"></script>
	<script src="https://cdn.jsdelivr.net/jquery.validation/1.15.1/jquery.validate.min.js"></script>
	<script src="https://kit.fontawesome.com/a076d05399.js"></script>
	<script src="JS/login-form-validation.js"></script>
	<script src="JS/mail.js"></script>

</head>
<body>
	<div class="w3-container w3-center">
		<div class="w3-card w3-shadow w3-center w3-third w3-padding-large" style="margin-left: 33.33%; margin-top: 5%;">
			<h3><b>Login <img src="https://img.icons8.com/cotton/24/000000/login-rounded-right--v2.png"/> </b></h3>
			<center>
			<hr style="width: 50%;">
			<form method="post" action="{{ url_for('welcome') }}">
			<?php include('errors.php');?>
  			<label class="w3-text" required>Email</label>
  			<input class="w3-input w3-border w3-margin-top" style="width: 80%;" name="email" id="email" type="text" required></p>
  			<label class="w3-text" required>Password</label>
  			<input class="w3-input w3-border w3-margin-top" style="width: 80%;" name="password" id="pass" type="password" required></p>
  			</center>
  			<p>Forgot Password? <a class="w3-text-teal" href="confirmOTP.php" style="text-decoration: none;">Reset.</a></p>
			<input type="submit" class="w3-btn w3-black" name="form2" value="Submit">
			<hr>
			<p>New User? <a class="w3-text-teal" href="userRegistration.php" style="text-decoration: none;">Signup.</a></p>

		</form>
		</div>
  	</div>
</body>
</html>
