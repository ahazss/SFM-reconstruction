let imgCount = 0;
window.addEventListener("load", () => {
    document.getElementById("file").onchange = uploadImg;
    document.getElementById("run").onclick = run;
    document.getElementById("download").style.display = "none";
    window.formData = new FormData();
    imgCount = 0;
});

function uploadImg() {
    //上传图片并预览
    if(imgCount+this.files.length>50){
        alert("请勿上传超过50张图片!");
    }
    for (let i = 0; i < this.files.length; ++i) {
        //每张图片大小不得超过1M
        if (this.files[i].size > 2097152) {
            alert("上传图片大小不得超过2M!");
            return false;
        }
    }

    if (imgCount === 0) {
        //第一次上传
        if (this.files.length === 1) {
            //只上传一张
            document.getElementById("pic1").src = window.URL.createObjectURL(this.files[0]);
            window.formData.append("pic[]", this.files[0]);
            imgCount += 1;
        }
        else {
            //上传2张或以上
            document.getElementById("pic1").src = window.URL.createObjectURL(this.files[0]);
            window.formData.append("pic[]", this.files[0]);
            imgCount += 1;
            document.getElementById("pic2").src = window.URL.createObjectURL(this.files[1]);
            window.formData.append("pic[]", this.files[1]);
            imgCount += 1;
            let list = document.getElementById("list");
            let run = document.getElementById("run");
            for (let i = 2; i < this.files.length; ++i) {
                let div = createDiv(window.URL.createObjectURL(this.files[i]), imgCount + 1);
                window.formData.append("pic[]", this.files[i]);
                imgCount += 1;
                list.insertBefore(div, run);
            }
        }
    }
    else if (imgCount === 1) {
        //已有一张图片
        if (this.files.length === 1) {
            //只上传一张
            document.getElementById("pic2").src = window.URL.createObjectURL(this.files[0]);
            window.formData.append("pic[]", this.files[0]);
            imgCount += 1;
        }
        else {
            //上传2张或以上
            document.getElementById("pic2").src = window.URL.createObjectURL(this.files[0]);
            window.formData.append("pic[]", this.files[0]);
            imgCount += 1;
            let list = document.getElementById("list");
            let run = document.getElementById("run");
            for (let i = 1; i < this.files.length; ++i) {
                let div = createDiv(window.URL.createObjectURL(this.files[i]), imgCount + 1);
                window.formData.append("pic[]", this.files[i]);
                imgCount += 1;
                list.insertBefore(div, run);
            }
        }
    }
    else {
        //已有两张及以上
        let list = document.getElementById("list");
        let run = document.getElementById("run");
        for (let i = 0; i < this.files.length; ++i) {
            let div = createDiv(window.URL.createObjectURL(this.files[i]), imgCount + 1);
            window.formData.append("pic[]", this.files[i]);
            imgCount += 1;
            list.insertBefore(div, run);
        }
    }
}

function createDiv(src, n) {
    let div = document.createElement("div");
    div.classList.add("preview");
    let img = document.createElement("img");
    img.src = src;
    img.id = "pic" + n;
    let p = document.createElement("p");
    p.classList.add("word");
    p.innerHTML = "图片" + n;
    div.appendChild(img);
    div.appendChild(p);
    return div;
}

function run() {
    if (imgCount < 2) {
        alert("请最少上传两张图片");
        return;
    }
    let xmlhttp;
    if (window.XMLHttpRequest) {
        // IE7+, Firefox, Chrome, Opera, Safari 浏览器执行的代码
        xmlhttp = new XMLHttpRequest();
    }
    else {
        //IE6, IE5 浏览器执行的代码
        xmlhttp = new ActiveXObject("Microsoft.XMLHTTP");
    }
    //请求成功时执行
    xmlhttp.onreadystatechange = function () {
        if (xmlhttp.readyState == 4 && xmlhttp.status == 200) {
            if (xmlhttp.responseText.indexOf("Bundle Adjustment")>0) {
                //成功
                 alert("运行成功，请点击下载");
                document.getElementById("run").style.display = "none";
                document.getElementById("download").style.display = "inline-block";
                
            }
            else {
               alert("运行失败!");
               document.getElementById("run").style.backgroundImage = "none";
               document.getElementById("runtext").innerHTML = "run";
            }
        }
    }
    window.formData.append("count", imgCount);
    xmlhttp.open("POST", "./php/main.php", true);
    xmlhttp.send(window.formData);
    alert("图片已上传，正在处理，请稍候！");
    document.getElementById("run").style.backgroundImage = "url(images/running.gif)";
    document.getElementById("runtext").innerHTML = "";
}