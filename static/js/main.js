const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("image");
const dropzoneContent = document.getElementById("dropzone-content");
const preview = document.getElementById("preview");
const previewImage = document.getElementById("preview-image");
const removeBtn = document.getElementById("remove-btn");
const clearBtn = document.getElementById("clear-btn");
const form = document.getElementById("upload-form");
const cameraInput = document.getElementById("camera-input");
const takePhotoBtn = document.getElementById("take-photo-btn");

// If the camera fallback swapped the "image" name onto #camera-input, any
// normal file pick must take it back, otherwise the pick posts no image field.
const restorePrimaryInput = () => {
  if (!fileInput.hasAttribute("name")) {
    fileInput.setAttribute("name", "image");
    fileInput.setAttribute("required", "required");
    cameraInput.removeAttribute("name");
  }
};

const setupDropzone = () => {
  dropzone.addEventListener("click", (e) => {
    if (e.target === removeBtn) return;
    fileInput.click();
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.remove("dragover");

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      restorePrimaryInput();
      fileInput.files = e.dataTransfer.files;
      handleFileSelect(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      restorePrimaryInput();
      handleFileSelect(e.target.files[0]);
    }
  });
};

const handleFileSelect = (file) => {
  if (!file.type.startsWith("image/")) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.src = e.target.result;
    dropzoneContent.classList.add("hidden");
    preview.classList.remove("hidden");
  };
  reader.readAsDataURL(file);
};

// Hands the captured file to the real #image input so /predict keeps seeing a
// single "image" field. The camera input carries no name for the same reason.
const attachCameraFile = (file) => {
  if (!file) return;

  if (typeof DataTransfer === "function") {
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;
    console.info("camera capture attached via DataTransfer");
  } else {
    // Safari < 14.1 cannot assign to input.files. Post the capture from the
    // camera input instead: move the name onto it and relax #image, which has
    // no file and would otherwise block submit.
    fileInput.removeAttribute("name");
    fileInput.removeAttribute("required");
    cameraInput.setAttribute("name", "image");
    console.warn("DataTransfer unavailable; posting from the camera input");
  }

  handleFileSelect(file);
};

let cameraOpening = false;

if (cameraInput && takePhotoBtn) {
  takePhotoBtn.addEventListener("click", (e) => {
    e.preventDefault();
    if (cameraOpening) return;
    cameraOpening = true;
    setTimeout(() => (cameraOpening = false), 1000);
    cameraInput.click();
  });

  cameraInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      attachCameraFile(e.target.files[0]);
    }
  });
}

const clearPreview = () => {
  fileInput.value = "";
  if (cameraInput) cameraInput.value = "";
  if (!fileInput.hasAttribute("name")) {
    fileInput.setAttribute("name", "image");
    cameraInput.removeAttribute("name");
  }
  fileInput.setAttribute("required", "required");
  previewImage.src = "";
  dropzoneContent.classList.remove("hidden");
  preview.classList.add("hidden");
};

removeBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  clearPreview();
});

clearBtn.addEventListener("click", (e) => {
  e.preventDefault();
  clearPreview();
  const resultCard = document.querySelector(".card.result");
  if (resultCard) {
    resultCard.remove();
  }
  window.scrollTo({ top: 0, behavior: "smooth" });
});

setupDropzone();