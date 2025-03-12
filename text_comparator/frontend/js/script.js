document.addEventListener("DOMContentLoaded", function () {
    enableDragAndDrop("drop-area1", "file1", "file1-label");
    enableDragAndDrop("drop-area2", "file2", "file2-label");

    document.getElementById("compare-button").addEventListener("click", compareFiles);
});

function enableDragAndDrop(dropAreaId, inputId, labelId) {
    let dropArea = document.getElementById(dropAreaId);
    let fileInput = document.getElementById(inputId);
    let fileLabel = document.getElementById(labelId);

    dropArea.addEventListener("click", () => fileInput.click());

    dropArea.addEventListener("dragover", (event) => {
        event.preventDefault();
        dropArea.classList.add("active");
    });

    dropArea.addEventListener("dragleave", () => {
        dropArea.classList.remove("active");
    });

    dropArea.addEventListener("drop", (event) => {
        event.preventDefault();
        dropArea.classList.remove("active");

        if (event.dataTransfer.files.length > 0) {
            fileInput.files = event.dataTransfer.files;
            fileLabel.textContent = event.dataTransfer.files[0].name;
        }
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            fileLabel.textContent = fileInput.files[0].name;
        }
    });
}

function compareFiles() {
    console.log("✅ כפתור Compare נלחץ!");

    let file1 = document.getElementById("file1").files[0];
    let file2 = document.getElementById("file2").files[0];

    if (!file1 || !file2) {
        alert("Please upload two text files.");
        console.log("⚠️ לא נבחרו כל הקבצים!");
        return;
    }

    let formData = new FormData();
    formData.append("file1", file1);
    formData.append("file2", file2);

    console.log("📤 שולח קבצים לשרת...");

    fetch("http://127.0.0.1:5000/upload", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log("📥 תשובה מהשרת:", data);
        displayDifferences(data);
    })
    .catch(error => console.error("❌ שגיאה בשליחת הבקשה:", error));
}

function displayDifferences(differences) {
    document.getElementById("text1").innerHTML = differences.text1;
    document.getElementById("text2").innerHTML = differences.text2;
}
