
function sendEmail() {
	console.log("inside send email function");
	var to =document.getElementById("emailreset");
	otp = generateOTP();
	Email.send({
	Host: "smtp.gmail.com",
	Username : "samruddhi.z@somaiya.edu",
	Password : "Samu@16091",
	To : to.value,
	From : "samruddhi.z@somaiya.edu",
	Subject : "Password Reset",
	Body : "We have received a password reset request from you. Here is your OTP (One Time Password): <b>" + otp + "</b> <br>Kindly do not share the OTP with anyone.",
	});

	document.getElementById("hiddenotp").value=otp;
}
		
function generateOTP() {
	var string = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'; 
	let OTP = ''; 
	var len = string.length; 
	for (let i = 0; i < 6; i++ ) { 
	    OTP += string[Math.floor(Math.random() * len)]; 
	}  
	return OTP;
}

