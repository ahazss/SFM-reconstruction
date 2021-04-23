export default function loadmodel(data, buildwithColor = true, UIon = false) {
    
    let Config = function () {
        this.pointSize = 0.01;
        this.pointColor = [255, 255, 255];
        this.showColor = true;
        this.backgroundColor = [0, 0, 0];
    }
    
    let scene, camera, renderer, clock, controls;
    let pointCloud, pointMaterial, geometry
    let config = new Config();
    init();
    createcloud(data);
    animate();

    function init() {
        // 初始化场景
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x000000);

        // 创建渲染器
        renderer = new THREE.WebGLRenderer({
            canvas: document.getElementById("container"),
            antialias: true, // 抗锯齿
            alpha: true
        });
        renderer.setSize(window.innerWidth, window.innerHeight);


        // 创建透视相机
        camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(0, 30, 20);
        camera.lookAt(0, 0, 0);

        controls = new THREE.OrbitControls(camera, renderer.domElement);

        // 环境光
        let ambientLight = new THREE.AmbientLight(0x606060);
        scene.add(ambientLight);
        // 平行光
        let directionalLight = new THREE.DirectionalLight(0xBCD2EE);
        directionalLight.position.set(1, 0.75, 0.5).normalize();
        scene.add(directionalLight);

        //xyz轴
        let axesHelper = new THREE.AxesHelper(30);
        scene.add(axesHelper);

        clock = new THREE.Clock();

        //GUI
        if (UIon) {
            let gui = new dat.GUI();
            let points = gui.addFolder("points");
            points.open();
            points.add(config, 'pointSize').min(0.2).max(0.5).step(0.01).onChange(value => {
                pointCloud.material.size = value;
            }).name("Size");
        }
    }

    function animate() {
        requestAnimationFrame(animate);
        let delta = clock.getDelta();
        controls.update(delta);
        renderer.render(scene, camera);
    }

    function createcloud(data) {
        if (data != null) {
            geometry = new THREE.Geometry(); //创建一个立方体几何对象Geometry
            //opencv 是右手坐标系  three.js是左手  会出现上下颠倒  因此取负值
            for (let i = 0; i < data.Points.length; ++i) {
                let p = new THREE.Vector3(data.Points[i][0] * 10, -data.Points[i][1] * 10, data.Points[i][2] * 10);
                geometry.vertices.push(p);
                let c = new THREE.Color(data.Colors[i][0] / 255, data.Colors[i][1] / 255, data.Colors[i][2] / 255);
                geometry.colors.push(c);
            }
            if (buildwithColor) {
                pointMaterial = new THREE.PointsMaterial({
                    size: 0.2,             //定义粒子的大小。默认为0.1
                    vertexColors: THREE.VertexColors,
                });
            }
            else {
                pointMaterial = new THREE.PointsMaterial({
                    size: 0.2,
                    colors: 0xffffff
                });
            }

            //生成点模型
            pointCloud = new THREE.Points(geometry, pointMaterial);
            //将模型添加到场景
            scene.add(pointCloud);
        }
    }
}