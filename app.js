// 完全ローカル参照（CDN禁止）
import vision from "./assets/mediapipe/tasks-vision/tasks-vision@0.10.3.js";
const { PoseLandmarker, FilesetResolver, DrawingUtils } = vision;

// ==== UI refs ====
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const octx = overlay.getContext("2d");

const onboarding = document.getElementById("onboarding");
const consentChk  = document.getElementById("consentChk");
const startBtn    = document.getElementById("startBtn");

const menuBtn = document.getElementById("menuBtn");
const drawer = document.getElementById("drawer");
const closeDrawer = document.getElementById("closeDrawer");

const cameraSelect = document.getElementById("cameraSelect");
const winSec = document.getElementById("winSec");
const winSecVal = document.getElementById("winSecVal");
const fps = document.getElementById("fps");
const fpsVal = document.getElementById("fpsVal");
const metricSel = document.getElementById("metric");
const metricLabel = document.getElementById("metricLabel");
const exportCsvBtn = document.getElementById("exportCsv");
const clearDataBtn = document.getElementById("clearData");

const barCanvas = document.getElementById("barChart");
const bctx = barCanvas.getContext("2d");
const varianceLabel = document.getElementById("varianceLabel");

// ==== 状態 ====
let stream = null;
let landmarker = null;
let running = false;
let lastTick = 0;
let lastUpdateAt = 0; // 最後に分散を追加した時間(ms)

// サンプル履歴: {t, pitch, yaw, roll}
const samples = [];
// 可視化用分散履歴（選択メトリクス）
const varianceSeries = [];

// ==== Onboarding ====
consentChk.addEventListener("change", ()=> startBtn.disabled = !consentChk.checked);
startBtn.addEventListener("click", async ()=>{
    onboarding.classList.remove("visible");
    await initApp();
});

menuBtn.onclick = ()=> drawer.classList.add("open");
closeDrawer.onclick = ()=> drawer.classList.remove("open");
winSec.oninput = ()=> winSecVal.textContent = winSec.value;
fps.oninput = ()=> fpsVal.textContent = fps.value;
metricSel.onchange = ()=> metricLabel.textContent = metricSel.value;

// ==== オーバービューの表示制御（スクロールで展開） ====
const overview = document.getElementById("overview");
function updateOverviewReveal(){
    if (!overview) return;
    if (window.scrollY > 10) overview.classList.add("expanded");
    else overview.classList.remove("expanded");
}
window.addEventListener("scroll", updateOverviewReveal, { passive: true });
// 初期状態を反映
updateOverviewReveal();

// ==== 追加: 更新周期UI ====
const updateIntervalInput = document.createElement("input");
updateIntervalInput.type = "range";
updateIntervalInput.min = 10;
updateIntervalInput.max = 60;
updateIntervalInput.step = 5;
updateIntervalInput.value = 30;
const updateLabel = document.createElement("span");
updateLabel.textContent = "30";

// #drawer 内に section が無い場合は生成してから追加
const drawerSection = document.querySelector("#drawer section") || (() => {
    const s = document.createElement("section");
    drawer.appendChild(s);
    return s;
})();

drawerSection.appendChild(document.createElement("hr"));
drawerSection.append("更新周期(秒): ", updateIntervalInput, updateLabel);

updateIntervalInput.oninput = () => updateLabel.textContent = updateIntervalInput.value;

// ==== カメラ ====
async function listCameras(){
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cams = devices.filter(d=>d.kind === "videoinput");
    cameraSelect.innerHTML = "";
    cams.forEach((c, i)=>{
        const opt = document.createElement("option");
        opt.value = c.deviceId;
        opt.textContent = c.label || `Camera ${i+1}`;
        cameraSelect.appendChild(opt);
    });
}
async function startCamera(deviceId){
    if (stream) stream.getTracks().forEach(t=>t.stop());
    stream = await navigator.mediaDevices.getUserMedia({
        video: { deviceId: deviceId? {exact: deviceId}: undefined, width: {ideal:1280}, height:{ideal:720} },
        audio: false
    });
    video.srcObject = stream;
    await new Promise(res=> video.onloadedmetadata = res);
    fitOverlayToVideo();
}
cameraSelect.onchange = ()=> startCamera(cameraSelect.value);
function fitOverlayToVideo(){
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
}

// ==== MediaPipe Pose 初期化（ローカルの wasm を使用）====
async function initPose(){
    const WASM_BASE = new URL("./assets/mediapipe/tasks-vision/wasm", import.meta.url).href;
    const files = await FilesetResolver.forVisionTasks(WASM_BASE);
    landmarker = await PoseLandmarker.createFromOptions(files, {
        baseOptions: { modelAssetPath: "assets/mediapipe/models/pose_landmarker_full.task", delegate: "GPU" },
        runningMode: "VIDEO", numPoses: 1
    });
}

// ==== 角度計算（Pythonの式をJSへ移植）====
// 入力は worldLandmarks（x,y,z）を期待。無ければ landmarks を使用（z=0扱い）。
function computeHeadAngles(nose, left_ear, right_ear){
    // いずれか欠損なら null
    if (!nose || !left_ear || !right_ear) return {pitch:null,yaw:null,roll:null};

    const dx = nose.x - (left_ear.x + right_ear.x)/2;
    const dy = nose.y - (left_ear.y + right_ear.y)/2;
    const dz = (nose.z ?? 0) - ((left_ear.z ?? 0) + (right_ear.z ?? 0))/2;

    const pitch = Math.atan2(dy, -dz) * 180/Math.PI; // X軸回り
    const yaw   = Math.atan2(dx, -dz) * 180/Math.PI; // Y軸回り

    const ear_dx = (right_ear.x - left_ear.x);
    const ear_dy = (right_ear.y - left_ear.y);
    const roll  = Math.atan2(ear_dy, -ear_dx) * 180/Math.PI; // Z軸回り

    return {pitch, yaw, roll};
}

// ==== 分散 ====
function variance(arr){
    const n = arr.length;
    if (!n) return 0;
    const m = arr.reduce((a,b)=>a+b,0)/n;
    return arr.reduce((a,b)=> a+(b-m)*(b-m), 0)/n;
}

// ==== 描画 ====
function drawOverlay(res){
    octx.clearRect(0,0,overlay.width,overlay.height);
    if (!res || !res.landmarks || !res.landmarks.length) return;
    const du = new DrawingUtils(octx);
    du.drawConnectors(res.landmarks[0], PoseLandmarker.POSE_CONNECTIONS, { color: "#0af" });
    du.drawLandmarks(res.landmarks[0], { color: "#0ff" });
}

function drawBars(series){
    const W = barCanvas.width, H = barCanvas.height;
    bctx.clearRect(0,0,W,H);

    const n = series.length;
    if (!n) return;

    const pad = 24;
    const cw = (W - pad*2) / n;
    const max = Math.max(...series, 0.0001);

    // x軸
    bctx.globalAlpha = .8;
    bctx.lineWidth = 1;
    bctx.strokeStyle = "rgba(255,255,255,.25)";
    bctx.beginPath(); bctx.moveTo(pad,H-pad); bctx.lineTo(W-pad,H-pad); bctx.stroke();

    // bars
    bctx.globalAlpha = 1;
    for (let i=0;i<n;i++){
        const v = series[i];
        const h = (H - pad*2) * (v/max);
        const x = pad + i*cw + 1;
        const y = H - pad - h;
        bctx.fillStyle = "rgba(125,211,252,0.65)";
        bctx.fillRect(x, y, Math.max(2, cw-2), h);
    }
}

// ==== ループ ====
async function tick(now){
    if (!running) return;

    const desired = 1000 / parseInt(fps.value,10);
    if (now - lastTick < desired){ requestAnimationFrame(tick); return; }
    lastTick = now;

    const res = await landmarker.detectForVideo(video, now);
    drawOverlay(res);

    // 角度抽出（world優先）
    let lm = null;
    if (res?.worldLandmarks?.length) lm = res.worldLandmarks[0];
    else if (res?.landmarks?.length){
        // 正規化座標→zは近似0で代用
        lm = res.landmarks[0].map(p=>({x:p.x, y:p.y, z:0}));
    }

    if (lm){
        // Pose index: 0:nose, 7:left_ear, 8:right_ear
        const nose = lm[0], left_ear = lm[7], right_ear = lm[8];
        const {pitch, yaw, roll} = computeHeadAngles(nose, left_ear, right_ear);
        const t = performance.now();
        samples.push({t, pitch, yaw, roll});

        // 1時間より古いサンプルは削除
        const cutoff = t - 3600*1000;
        while (samples.length && samples[0].t < cutoff) samples.shift();

        // 更新周期ごとに分散を追加
        const intervalMs = parseInt(updateIntervalInput.value,10)*1000;
        if (t - lastUpdateAt >= intervalMs){
            const key = metricSel.value;
            const vals = samples.map(s=> s[key]).filter(v=> v != null && isFinite(v));
            const v = variance(vals);
            varianceSeries.push(v);

            // 1時間分だけ保持
            const maxPoints = Math.ceil(3600 / (intervalMs/1000));
            while (varianceSeries.length > maxPoints) varianceSeries.shift();

            varianceLabel.textContent = v.toFixed(4);
            drawBars(varianceSeries);

            lastUpdateAt = t; // 更新時刻を記録
        }
    }

    requestAnimationFrame(tick);
}

// ==== CSV出力（ブラウザで生成）====
exportCsvBtn.onclick = ()=>{
    // t,pitch,yaw,roll, selectedVariance（対になる長さで出す）
    const len = samples.length;
    const rows = [["timestamp_ms","pitch_deg","yaw_deg","roll_deg","window_variance("+metricSel.value+")"]];
    const startIdx = Math.max(0, varianceSeries.length ? len - varianceSeries.length : 0);
    for(let i=0;i<len;i++){
        const s = samples[i];
        const varVal = varianceSeries[i - startIdx] ?? "";
        rows.push([Math.round(s.t), fix(s.pitch), fix(s.yaw), fix(s.roll), (typeof varVal==="number"? varVal.toFixed(6):"")]);
    }
    const csv = rows.map(r=>r.join(",")).join("\n");
    downloadBlob(new Blob([csv],{type:"text/csv;charset=utf-8"}), `head_sway_pose_${Date.now()}.csv`);
};
clearDataBtn.onclick = ()=>{
    samples.length = 0;
    varianceSeries.length = 0;
    varianceLabel.textContent = "–";
    drawBars([]);
};
const fix = v => (v==null || !isFinite(v)) ? "" : (+v).toFixed(6);
function downloadBlob(blob, name){
    const a=document.createElement("a"); a.href=URL.createObjectURL(blob); a.download=name; a.click();
    setTimeout(()=>URL.revokeObjectURL(a.href), 10000);
}

// ==== 初期化 ====
async function initApp(){
    await listCameras();
    await startCamera(cameraSelect.value || undefined);
    await initPose();

    running = true;
    requestAnimationFrame(tick);
    window.addEventListener("resize", fitOverlayToVideo);
}

// 起動時は同意画面
onboarding.classList.add("visible");