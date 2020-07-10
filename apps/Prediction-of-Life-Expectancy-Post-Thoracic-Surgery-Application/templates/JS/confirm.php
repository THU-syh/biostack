<?php include('server.php');?>
<!DOCTYPE HTML>
<html>
<head>
	<title>User Registration</title>
	<link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
	<script src="https://smtpjs.com/v3/smtp.js"></script>
	<script src="JS/mail.js"></script>
	
</head>
<body>
	<div class="w3-container w3-center">
		<div class="w3-card w3-shadow w3-center w3-third w3-padding-large" style="margin-left: 33.33%; margin-top: 5%;">
			<h3><b>Reset Password</b></h3>
			<?php include('errors.php');?>
			<center>
			<hr style="width: 50%;">
		  	<form method="post" action="confirm.php"> 
  			<label class="w3-text" >Enter the OTP sent on your registered email id</label>
  			<input class="w3-input w3-border w3-margin-top" style="width: 80%;" name="onetimepassword" id="onetimepassword" type="text" required></p>
  			<input type="submit" class="w3-btn w3-black" value="Submit" name="form4">
  			</form>
  		
			<hr>
			<p> <a class="w3-text-teal" href="login.php" style="text-decoration: none;">Login</a> &nbsp <a class="w3-text-teal" href="userRegistration.php" style="text-decoration: none;">Signup</a></p>
		</center>
		</div>
  	</div>
</body>
</html>