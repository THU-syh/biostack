<?php include('server.php');?>
<!DOCTYPE HTML>
<html>
<head>
	<title>Password Reset</title>
	<link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
</head>
<body>
	<div class="w3-container w3-center">
		<div class="w3-card w3-shadow w3-center w3-third w3-padding-large" style="margin-left: 33.33%; margin-top: 5%;">

			<h3><b>Reset Password</b></h3>
			<center>
			<hr style="width: 50%;">
			<!--<form action="action.php"> </form>-->
			<form method="post" action="login.php">
			<?php include('errors.php');?>
  			<label class="w3-text" >New Password</label>
  			<input class="w3-input w3-border w3-margin-top" style="width: 80%;" name="newPassword" type="password" required></p>
  			<label class="w3-text" >Confirm Password</label>
  			<input class="w3-input w3-border w3-margin-top" style="width: 80%;" name="confnewpassword" type="password" required></p>
  			</center>
  			<input type="submit" name="form5" class="w3-btn w3-black" value="Reset">
			<hr>
			</form>
			<p> <a class="w3-text-teal" href="login.html" style="text-decoration: none;">Login</a> &nbsp <a class="w3-text-teal" href="userRegistration.html" style="text-decoration: none;">Signup</a></p>
		</div>
  	</div>
</body>
</html>