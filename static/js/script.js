document.addEventListener("DOMContentLoaded", function () {
  const predictionForm = document.getElementById("prediction-form");

  // --- Slider Value Display Logic ---
  const sliders = document.querySelectorAll('input[type="range"]');
  sliders.forEach((slider) => {
    const valueDisplay = document.getElementById(slider.dataset.display);
    if (valueDisplay) {
      valueDisplay.textContent = slider.value;
      slider.addEventListener("input", () => {
        valueDisplay.textContent = slider.value;
      });
    }
  });

  // --- Function to Save/Load Form Data ---
  function saveFormData() {
    const formData = new FormData(predictionForm);
    const data = Object.fromEntries(formData.entries());
    sessionStorage.setItem("hotelFormData", JSON.stringify(data));
  }

  function loadFormData() {
    const savedData = sessionStorage.getItem("hotelFormData");
    if (savedData) {
      const data = JSON.parse(savedData);
      for (const key in data) {
        const input = predictionForm.elements[key];
        if (input) {
          if (input.type === "range") {
            input.value = data[key];
            const display = document.getElementById(input.dataset.display);
            if (display) display.textContent = data[key];
          } else {
            input.value = data[key];
          }
        }
      }
    }
  }

  if (predictionForm) {
    Array.from(predictionForm.elements).forEach((element) => {
      element.addEventListener("change", saveFormData);
    });
  }
  loadFormData();

  // --- Prediction Form Submission Logic ---
  if (predictionForm) {
    predictionForm.addEventListener("submit", function (e) {
      e.preventDefault();
      saveFormData();

      const formData = new FormData(predictionForm);
      const data = Object.fromEntries(formData.entries());

      for (const key in data) {
        if (!isNaN(data[key]) && data[key] !== "") {
          data[key] = Number(data[key]);
        }
      }

      const resultBox = document.getElementById("result-box");
      const resultImg = document.getElementById("result-img");
      const resultText = document.getElementById("result-text");
      const resultProba = document.getElementById("result-proba");
      resultBox.style.display = "block";
      resultText.textContent = "Analyzing...";
      resultProba.textContent = "Please wait";
      resultImg.src = "";

      fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      })
        .then((response) => response.json())
        .then((result) => {
          if (result.error) {
            resultText.innerText = `Error: ${result.error}`;
            resultProba.innerText = "";
            resultBox.className = "result-box";
          } else {
            resultText.innerText = result.prediction_text;
            resultProba.innerText = `Confidence: ${result.probability}`;

            // --- BUG FIX IS HERE ---
            // Using a more specific check for the cancellation text.
            if (result.prediction_text === "Booking will be Cancelled") {
              resultBox.className = "result-box will-cancel";
              resultImg.src = "/static/images/close.png";
            } else {
              resultBox.className = "result-box will-not-cancel";
              resultImg.src = "/static/images/checkbox.png";
            }
          }
        })
        .catch((error) => {
          console.error("Error:", error);
          resultText.innerText = "An error occurred. Please try again.";
          resultProba.innerText = "";
        });
    });
  }

  // --- Charting Logic for Trends Page (No changes needed here) ---
  const monthlyDataEl = document.getElementById("monthlyData");
  if (monthlyDataEl) {
    // ... (rest of the script remains the same)
  }

  const weeklyDataEl = document.getElementById("weeklyData");
  if (weeklyDataEl) {
    // ... (rest of the script remains the same)
  }
});
