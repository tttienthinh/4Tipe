const canvas = document.getElementById("canvas")
const result = document.getElementById("Result")

canvas.addEventListener("mousedown", start, false)
canvas.addEventListener("mousemove", draw, false)
canvas.addEventListener("mouseup", stop, false)
canvas.addEventListener("mouseout", stop, false)

// Pour le canvas
let context = canvas.getContext("2d")

let is_drawing = false;
context.fillStyle = 'white';
context.fillRect(0, 0, canvas.width, canvas.height);

context.strokeStyle = "black";
context.lineWidth = 4;
context.lineCap = "round";
context.lineJoin = "round";


function start(event) {
    is_drawing = true;
    context.beginPath()
    context.moveTo(
        event.clientX - canvas.offsetLeft,
        event.clientY - canvas.offsetTop,
    )
    event.preventDefault();
}

function draw(event) {
    if ( is_drawing ) {
        context.lineTo(
            event.clientX - canvas.offsetLeft,
            event.clientY - canvas.offsetTop, 
        );
        context.stroke();
    }
    event.preventDefault();
}

function stop(event) {
    if ( is_drawing ) {
        context.stroke();
        context.closePath();
        is_drawing = false
    }
    event.preventDefault();
}

// Pour le boutton effacer
function effacer() {
    console.log("Effacer");
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = 'white';
    context.fillRect(0, 0, canvas.width, canvas.height);
    result.innerHTML = '';
}

// Pour prédire
function dataURLToBlob(dataURL) {
    // Code taken from https://github.com/ebidel/filer.js
    var parts = dataURL.split(';base64,');
    var contentType = parts[0].split(":")[1];
    var raw = window.atob(parts[1]);
    var rawLength = raw.length;
    var uInt8Array = new Uint8Array(rawLength);
  
    for (var i = 0; i < rawLength; ++i) {
      uInt8Array[i] = raw.charCodeAt(i);
    }
    return new Blob([uInt8Array], { type: contentType });
}
function predire() {
    var data = dataURLToBlob(canvas.toDataURL());
    console.log(version);
    console.log(data);
    $.ajax({
        type: 'POST',
        url: 'prediction/' + version,
        data: data,
        processData: false,
        contentType: false
    }).done(function (data) {

        var json = jQuery.parseJSON(data);
        result.innerHTML = '<h2 class="alert-heading">'+ json.message +'</h2>\n';

    }).fail(function (data) {
        console.log('Fail!');
    });
}