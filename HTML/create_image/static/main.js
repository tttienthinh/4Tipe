const canvas = document.getElementById("canvas")
const clear = document.getElementById("ClearButton")
const predict = document.getElementById("PredictButton")
const result = document.getElementById("result")

canvas.addEventListener("mousedown", start, false)
canvas.addEventListener("mousemove", draw, false)
canvas.addEventListener("mouseup", stop, false)
canvas.addEventListener("mouseout", stop, false)

clear.addEventListener("click", effacer)
predict.addEventListener("click", predire)

// Pour le canvas
let context = canvas.getContext("2d")

let is_drawing = false;
context.fillStyle = 'white';
context.fillRect(0, 0, canvas.width, canvas.height);

context.strokeStyle = "black";
context.lineWidth = 10;
context.lineCap = "round";
context.lineJoin = "round";

let clientX = 0;
let clientY = 0;

var values = [2, 4, 8, 16, 32]
var index = [0, 1, 2, 3, 4]
var contextList = [];
values.forEach(list_create);
function list_create(value) {
    let ctx = document.getElementById(value).getContext("2d");
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, document.getElementById(value).width, document.getElementById(value).height);

    ctx.strokeStyle = "black";
    ctx.lineWidth = 10;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    contextList.push([document.getElementById(value), document.getElementById(value).getContext("2d")])
}

function list_start(value) {
    contextList[value][1].beginPath()
    contextList[value][1].moveTo(
        clientX - contextList[value][0].offsetLeft,
        clientY - contextList[value][0].offsetTop,
    );
}
function list_draw(value) {
    contextList[value][1].lineTo(
        clientX - contextList[value][0].offsetLeft,
        clientY - contextList[value][0].offsetTop,
    );
    contextList[value][1].stroke();
}
function list_stop(value) {
    contextList[value][1].stroke();
    contextList[value][1].closePath();
}
function list_effacer(value) {
    contextList[value][1].clearRect(0, 0, canvas.width, canvas.height);
    contextList[value][1].fillStyle = 'white';
    contextList[value][1].fillRect(0, 0, canvas.width, canvas.height);
}

function start(event) {
    is_drawing = true;
    context.beginPath()
    clientX = event.clientX
    clientY = event.clientY
    context.moveTo(
        event.clientX - canvas.offsetLeft,
        event.clientY - canvas.offsetTop,
    )
    index.forEach(list_start);
    event.preventDefault();
}

function draw(event) {
    if ( is_drawing ) {
        clientX = event.clientX
        clientY = event.clientY

        context.lineTo(
            event.clientX - canvas.offsetLeft,
            event.clientY - canvas.offsetTop, 
        );
        context.stroke();
        index.forEach(list_draw);
    }
    event.preventDefault();
}

function stop(event) {
    if ( is_drawing ) {
        context.stroke();
        context.closePath();
        is_drawing = false
        index.forEach(list_stop);
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
    index.forEach(list_effacer);
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
    console.log("test");
    console.log(data);
    $.ajax({
        type: 'POST',
        url: '/recognize',
        data: data,
        processData: false,
        contentType: false
    }).done(function (data) {

        var json = jQuery.parseJSON(data);
        result.innerHTML = '<h2 class="alert-heading">Result: '+json.num+'</25>\n';

    }).fail(function (data) {
        console.log('Fail!');
    });
}