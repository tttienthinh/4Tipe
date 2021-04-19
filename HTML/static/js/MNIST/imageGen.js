const canvas = document.getElementById("canvas")
const clear = document.getElementById("ClearButton")
const predict = document.getElementById("PredictButton")
const log = document.getElementById("Log")
const consigne = document.getElementById("Consigne")

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
    ctx.lineWidth = value;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    contextList.push([document.getElementById(value), document.getElementById(value).getContext("2d")])
}

function list_start(value) {
    console.log(clientX)
    contextList[value][1].beginPath()
    contextList[value][1].moveTo(
        clientX - canvas.offsetLeft,
        clientY - canvas.offsetTop,
    );
}
function list_draw(value) {
    contextList[value][1].lineTo(
        clientX - canvas.offsetLeft,
        clientY - canvas.offsetTop,
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
function list_url(contextList) {
    let url_list = [];
    for ( let i=0; i<contextList.length; i++) {
        url_list.push(dataURLToBlob(contextList[i][0].toDataURL()));
    };
    return url_list
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
    index.forEach(list_effacer);
    log.innerHTML = '<h2 class="alert-heading">Effacement du Canvas</h2>\n';
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
function enregistrer() {
    var list_data = list_url(contextList);
    var data = dataURLToBlob(canvas.toDataURL());
    console.log("test");
    console.log(data);
    for ( let i=0; i<contextList.length; i++) {
        $.ajax({
            type: 'POST',
            url: '/MNIST/enregistrer/'+i,
            data: list_data[i],
            processData: false,
            contentType: false
        }).done(function (data) {
        }).fail(function (data) {
            console.log('Fail!');
        });
    };
    $.ajax({
        type: 'GET',
        url: '/MNIST/enregistrer/0',
        processData: false,
        contentType: false
    }).done(function (data) {
        var json = jQuery.parseJSON(data);
        effacer();
        log.innerHTML = '<h2 class="alert-heading">Bien enregistré</h2>\n';
        consigne.innerHTML = 'Dessine un <b>' + json.nombre + '</b> ou <b>Télécharge</b> si tu as fini.'
    }).fail(function (data) {
        console.log('Fail!');
    });    
}

function telecharger() {
    console.log("télechargement");
    log.innerHTML = '<h2 class="alert-heading">Bien téléchargé au format zip</25>\n';
}

function effacer_fichier() {
    effacer()
    $.ajax({
        type: 'GET',
        url: '/MNIST/effacer_fichier',
    }).done(function (data) {
        var json = jQuery.parseJSON(data);
        log.innerHTML = '<h2 class="alert-heading">Fichiers bien effacés</h2>\n';
        consigne.innerHTML = 'Dessine un <b>' + json.nombre + '</b> ou <b>Télécharge</b> si tu as fini.'
    }).fail(function (data) {
        console.log('Fail!');
    });
}