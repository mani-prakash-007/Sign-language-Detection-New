document.addEventListener('DOMContentLoaded', function() {
  const captureBtn = document.getElementById('capture-btn');
  const signResult = document.getElementById('sign-result');
  const confidenceResult = document.getElementById('confidence-result');
  
  captureBtn.addEventListener('click', function() {
      // Show loading state
      signResult.textContent = 'Processing...';
      confidenceResult.textContent = '';
      
      // Send request to server for prediction
      fetch('/predict', {
          method: 'POST',
          headers: {
              'Content-Type': 'application/json',
          }
      })
      .then(response => response.json())
      .then(data => {
          if (data.error) {
              signResult.textContent = 'Error: ' + data.error;
              confidenceResult.textContent = '';
          } else {
              signResult.textContent = data.sign;
              confidenceResult.textContent = `Confidence: ${data.confidence.toFixed(2)}%`;
          }
      })
      .catch(error => {
          console.error('Error:', error);
          signResult.textContent = 'Error making prediction';
          confidenceResult.textContent = '';
      });
  });
});