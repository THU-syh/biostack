<?php include 'config.php';
require_once 'declare_globals.php';

session_start();

//REGISTRATION FORM SERVER
if(isset($_POST['form1'])){
	//get input from the form also removing the special characters if any

	$username= $_POST['username'];
	$email= $_POST['email'];
  $password= $_POST['pass'];
	$confirmPass= $_POST['confpass'];

	// first check the database to make sure a user does not already exist with the same username and/or email
  	$user_check_query = "SELECT * FROM user WHERE name='$username' OR email='$email' LIMIT 1";
  	$result = mysqli_query($con, $user_check_query);
  	$user = mysqli_fetch_assoc($result);
  	// if user exists
  	if ($user) { 
      if($user['email'] === $email){
        $flag=1;
        header("refresh:2; url=login.php" ); 
        array_push($errors, "This email is already registered. Kindly login. Redirecting to login page...");
      }
  	}

   if($flag==0){
      // Finally, register user if there are no errors in the form
    $passmd = md5($password);//encrypt the password before saving in the database

    $query = "INSERT INTO user (name, email, password) VALUES('$username', '$email', '$passmd')";
   
    if(mysqli_query($con, $query)){
      $_SESSION['username'] = $username;
      $_SESSION['email'] = $email;
      $_SESSION['success'] = "You are now logged in";
    
      header("Location: welcome.php");
    }
    else{
      array_push($errors, "Did not register you huh");
    }


    

    }

  
}

//LOGIN FORM SERVER
else if(isset($_POST['form2'])){
	
	$email= $_POST['email'];
  $password= $_POST['pass'];

  $user_check_query = "SELECT * FROM user WHERE email='$email' LIMIT 1";
  $result = mysqli_query($con, $user_check_query);
  $user = mysqli_fetch_assoc($result);
  	// if user exists
  if ($user) { 
   	if ($user['email'] === $email && $user['password'] === md5($password)) {

        $_SESSION['username'] =$user['name'];
        $_SESSION['email'] = $user['email'];
        $_SESSION['success'] = "You are now logged in";
     		header('Location: welcome.php');

    	}
    	else {
  		array_push($errors, "Wrong username/password combination");
  		}
    	
  	}
    else{
      array_push($errors, "User does not exist. Kindly Register.");
    }
}

//SEND OTP FORM SERVER
else if(isset($_POST['form3'])){
	$abc="";
	$resetemail="";
	$resetemail=$_POST['emailreset'];
  setcookie("reset_email", $resetemail, time()+30*24*60*60);
	// Function to generate OTP 
	function generateNumericOTP($n) { 
      
    // Take a generator string which consist of 
    // all numeric digits 
    $generator = "1357902468"; 
  
    // Iterate for n-times and pick a single character 
    // from generator and append it to $result 
      
    // Login for generating a random character from generator 
    //     ---generate a random number 
    //     ---take modulus of same with length of generator (say i) 
    //     ---append the character at place (i) from generator to result 
  
    $result = ""; 
  
    for ($i = 1; $i <= $n; $i++) { 
        $result .= substr($generator, (rand()%(strlen($generator))), 1); 
    } 
  
    // Return result 
    return $result; 
	} 
  
	// Main program 
	$n = 6; 
	$otp = generateNumericOTP($n);
	$GLOBALS['sentotp']=$otp;
	$to = $resetemail;
  // first check the database to make sure a user does not already exist with the same username and/or email
    $user_check_query = "SELECT * FROM user WHERE email='$resetemail' LIMIT 1";
    $result = mysqli_query($con, $user_check_query);
    $user = mysqli_fetch_assoc($result);
    // if user exists
    if ($user) { 
      $sql = "UPDATE user SET otp='$sentotp' WHERE email='$resetemail'";

      if ($con->query($sql) === TRUE) {
        //Send OTP to registered email address.
         $subject = "Password Reset Request";
         
         $message = "We have received a password reset request from you. Here is your OTP (One Time Password): <b>".$otp."</b> <br>Kindly do not share the OTP with anyone.";

         $header = "From:samruddhi.z@somaiya.edu \r\n";
         // $header .= "Cc:afgh@somedomain.com \r\n";
         $header .= "MIME-Version: 1.0\r\n";
         $header .= "Content-type: text/html\r\n";
         
         $retval = mail($to,$subject,$message,$header);


         
         if( $retval == true ) {
            header("Location: confirm.php");
         }else {
            array_push($errors, "Message could not be sent...");
         }
      } 
      else {
          echo "Error. Please try again. " . $conn->error;
      }
    }
    else{
      array_push($errors, "User does not exist. Kindly enter registered email address.");
    }
         	
}

//OTP CONFIRMATION FORM SERVER
else if(isset($_POST['form4'])){

	$enteredotp=$_POST['onetimepassword'];
  //echo "Entered otp is".$enteredotp;
  
  if(isset($_COOKIE["reset_email"])){
    $resetemail = $_COOKIE["reset_email"];
    //echo "Request email id is".$resetemail;
} else{
    array_push($errors, "Session Expired. Try again.");
    header( "refresh:2; url=confirmOTP.php" );
}
	//Get otp from database to match with entered otp
    $user_check_query = "SELECT otp FROM user WHERE email='$resetemail' LIMIT 1";
    $result = mysqli_query($con, $user_check_query);
    $user = mysqli_fetch_assoc($result);
    // if user exists
    if ($user) { 
          //array_push($errors, "User found in the database");
          // echo "saved otp is".$user['otp'];
          if($enteredotp == $user['otp']){
            header("Location: resetPassword.php");
          }
          else{
            array_push($errors, "Wrong OTP");
          }
    }


	// if($sentotp==$enteredotp){
	// 	array_push($errors, "Correct otp!" ) ;
	// }
	// else{
	// 	array_push($errors, "Wrong OTP!");
	// }
  //include 'errors.php';
}

//RESET TO NEW PASSWORD SERVER
else if(isset($_POST['form5'])){

  $passwordreset = $_POST['newPassword'];
  $confpasswordreset = $_POST['confnewpassword'];

  if($passwordreset == $confpasswordreset){

    //get email who requested password reset
    if(isset($_COOKIE["reset_email"])){
      $resetemail = $_COOKIE["reset_email"];
    
      //Change the password into the database
    
      $passmd = md5($passwordreset);
      $query = "UPDATE user SET password='$passmd' WHERE email='$resetemail'";
    
      if (mysqli_query($con, $query)){
          array_push($errors, "Password reset successful! Kinly Login.") ;
          header("refresh:2; url=login.php" );
          
          //Delete cookies
          setcookie("reset_email", "", time()-3600);
      } 
      else {
          array_push($errors, "Reset Failed.");
      }

    } 
    else{
      array_push($errors, "Session Expired. Try again.");
      header("refresh:2; url=confirmOTP.php" );
    }
    
  }
  else {
    array_push($errors, "Passwords do not match!");
  }

}

?>