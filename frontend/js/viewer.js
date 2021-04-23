import loadmodel from "./loadmodel.js";
//初步开发的版本  健壮性贼差
window.addEventListener("load", onload);

function onload() {
    document.getElementById("uploadfile").onchange = changeFileName;
    document.getElementById("build").onclick = buildDemo;
    document.getElementById("buildwithColor").onclick = buildDemo;

}

function changeFileName() {
    document.getElementById("filename").innerHTML = document.getElementById("uploadfile").value;
}
//将用户选中的json上传
function buildDemo(event) {
    let input = document.getElementById("uploadfile");
    let file = input.files[0];
    if (file === undefined) {
        return;
    }
    document.getElementById("upload").style.display = "none";
    document.getElementById("container").style.display = "normal";
    let reader = new FileReader();
    reader.readAsText(file);
    if (event.currentTarget.id === "build") {
        reader.onload = function (f) {
            let data = JSON.parse(this.result);
            loadmodel(data, false, true);
        }
    }
    else {
        reader.onload = function (f) {
            let data = JSON.parse(this.result);
            loadmodel(data, true, true);
        }
    }
}