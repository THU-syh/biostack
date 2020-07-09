

 $(function() {
  $("form[id='Registration']").validate({
    
    errorClass: 'w3-text-teal',
    // Specify validation rules
    rules: {
      // The key name on the left side is the name attribute
      // of an input field. Validation rules are defined
      // on the right side
      "username" : {required: true},
      "email" : {
        required: true,
        // Specify that email should be validated
        // by the built-in "email" rule
        email: true
      },
      "pass" : {
        required: true,
        minlength: 4 
      },
      "confpass" : {
        required: true,
        minlength: 4,
        equalTo: "#pass"
      }
    },
    // Specify validation error messages
    messages: {
      "username": "Please enter your name.",
      "email": {
        required: "This field is required.",
        email:"Please enter a valid email address."
      },
      "pass": {
        required: "Please provide a password.",
        minlength: "Your password must be at least 4 characters long."
      },
      "confpass": {
        required: "Please provide a password.",
        minlength: "Your password must be at least 4 characters long.",
        equalTo: "Your Passwords Must Match"
      }
      
      
    },
    // Make sure the form is submitted to the destination defined
    // in the "action" attribute of the form when valid
    submitHandler: function(form) {
      form.submit();
    }
  });
});
