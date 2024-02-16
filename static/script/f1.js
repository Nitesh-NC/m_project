// document.getElementById('fileInput').addEventListener('change', function () {
//     const previewContainer = document.getElementById('previewContainer');
//     const previewImage = document.getElementById('previewImage');
//     const submitButton = document.getElementById('submitButton');
//     const uploadMessage = document.getElementById('uploadMessage');
//     const fileInput = this;

//     if (fileInput.files && fileInput.files[0]) {
//         const reader = new FileReader();

//         reader.onload = function (e) {
//             previewImage.src = e.target.result;
//             previewContainer.style.display = 'block';
//             submitButton.classList.remove('hidden');
//             uploadMessage.classList.remove('hidden');
//         };

//         reader.readAsDataURL(fileInput.files[0]);
//     }
// });


document.getElementById('fileInput').addEventListener('change', function () {
    const previewContainer = document.getElementById('previewContainer');
    const previewImage = document.getElementById('previewImage');
    const submitButton = document.getElementById('submitButton');
    const uploadMessage = document.getElementById('uploadMessage');
    const fileInput = this;

    if (fileInput.files && fileInput.files[0]) {
        const reader = new FileReader();

        reader.onload = function (e) {
            previewImage.src = e.target.result;
            previewContainer.style.display = 'block';
            submitButton.classList.remove('hidden');
            uploadMessage.classList.remove('hidden');
        };

        reader.readAsDataURL(fileInput.files[0]);
    }
});
