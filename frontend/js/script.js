"use strict";

const form = document.getElementById("predictionForm");
const imageInput = document.getElementById("imageInput");
const dropZone = document.getElementById("dropZone");
const previewWrap = document.getElementById("previewWrap");
const preview = document.getElementById("imagePreview");
const detectButton = document.getElementById("detectButton");
const resetButton = document.getElementById("resetButton");
const removeImage = document.getElementById("removeImage");
const formError = document.getElementById("formError");
const resultSection = document.getElementById("resultSection");

function showError(message) {
  formError.textContent = message;
  formError.classList.remove("d-none");
}

function clearError() {
  formError.textContent = "";
  formError.classList.add("d-none");
}

function setLoading(isLoading) {
  detectButton.disabled = isLoading || !imageInput.files.length;
  detectButton.querySelector(".button-text").textContent = isLoading
    ? "Analysing image..."
    : "Detect disease";
  detectButton.querySelector(".spinner-border").classList.toggle("d-none", !isLoading);
}

function validateAndPreview(file) {
  clearError();
  resultSection.classList.add("d-none");
  if (!file) return;
  const acceptedTypes = ["image/jpeg", "image/png", "image/webp"];
  if (!acceptedTypes.includes(file.type)) {
    showError("Please select a PNG, JPG, JPEG, or WEBP image.");
    imageInput.value = "";
    return;
  }
  if (file.size > 8 * 1024 * 1024) {
    showError("Please select an image smaller than 8 MB.");
    imageInput.value = "";
    return;
  }
  preview.src = URL.createObjectURL(file);
  previewWrap.classList.remove("d-none");
  detectButton.disabled = false;
}

function resetForm() {
  form.reset();
  preview.removeAttribute("src");
  previewWrap.classList.add("d-none");
  resultSection.classList.add("d-none");
  clearError();
  setLoading(false);
}

function displayResult(data) {
  document.getElementById("plantName").textContent = data.plant;
  document.getElementById("diseaseName").textContent = data.disease;
  document.getElementById("diseaseDetail").textContent = data.disease;
  document.getElementById("confidenceValue").textContent = `${data.confidence}%`;
  document.getElementById("description").textContent = data.description;
  document.getElementById("causes").textContent = data.causes;
  document.getElementById("prevention").textContent = data.prevention;
  document.getElementById("treatment").textContent = data.treatment;
  resultSection.classList.remove("d-none");
  resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

imageInput.addEventListener("change", () => validateAndPreview(imageInput.files[0]));
resetButton.addEventListener("click", resetForm);
removeImage.addEventListener("click", resetForm);

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("drag-over");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("drag-over");
  });
});
dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  imageInput.files = transfer.files;
  validateAndPreview(file);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = imageInput.files[0];
  if (!file) {
    showError("Choose a leaf image before detecting disease.");
    return;
  }
  setLoading(true);
  clearError();
  try {
    const body = new FormData();
    body.append("image", file);
    const response = await fetch("/predict", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Unable to analyse this image.");
    displayResult(data);
  } catch (error) {
    showError(error.message || "Network error. Please try again.");
  } finally {
    setLoading(false);
  }
});
