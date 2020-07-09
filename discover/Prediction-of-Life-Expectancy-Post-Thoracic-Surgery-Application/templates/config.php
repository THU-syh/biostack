<?php
/* Database credentials.*/
define('DB_SERVER', 'localhost');
define('DB_USERNAME', 'root');
define('DB_PASSWORD', '');
define('DB_NAME', 'thoracic');
 
/*Connect to MySQL database */
$con = mysqli_connect(DB_SERVER, DB_USERNAME, DB_PASSWORD, DB_NAME);
 
// Check connection
if($con === false){
    die("ERROR: Could not connect. " . mysqli_connect_error());
}
/*else{
	echo "successfull!";
}*/
?>

<!-- Below Change In php.ini File

	Mail configuration
	https://stackoverflow.com/questions/15965376/how-to-configure-xampp-to-send-mail-from-localhost

SMTP=smtp.gmail.com 
smtp_port=587 
sendmail_from = your@gmail.com 
sendmail_path = "\"C:\xampp\sendmail\sendmail.exe\" -t" 
extension=php_openssl.dll 
Below Change In sendmail.ini File

smtp_server=smtp.gmail.com
smtp_port=587
error_logfile=error.log
debug_logfile=debug.log 
auth_username=yourmail@gmail.com 
auth_password=your-gmail-password 
force_sender=yourmail@gmail.com   -->