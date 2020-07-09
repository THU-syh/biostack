

 $(function() {
  $("form[id='login']").validate({
    
    errorClass: 'w3-text-teal',
    // Specify validation rules
    rules: {
      // The key name on the left side is the name attribute
      // of an input field. Validation rules are defined
      // on the right side
    
      "email" : {
        required: true
      },
      "pass" : {
        required: true
      }
    },
    // Specify validation error messages
    messages: {
      "email": {
        required: "Please provide registered email address."
      },
      "pass": {
        required: "Please provide a password."
      }     
      
    },
    // Make sure the form is submitted to the destination defined
    // in the "action" attribute of the form when valid
    submitHandler: function(form) {
      form.submit();
    }
  });
});
