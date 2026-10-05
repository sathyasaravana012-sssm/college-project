const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("image");
const dropzoneContent = document.getElementById("dropzone-content");
const preview = document.getElementById("preview");
const previewImage = document.getElementById("preview-image");
const removeBtn = document.getElementById("remove-btn");
const clearBtn = document.getElementById("clear-btn");
const form = document.getElementById("upload-form");

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
      fileInput.files = e.dataTransfer.files;
      handleFileSelect(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
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

const clearPreview = () => {
  fileInput.value = "";
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