/* 校验网页里的前向计算是否和 mnist_cnn.py 一致。
 *
 * 做法：trace_cnn.py 导出时用 numpy 算好了「每个快照 × 每张探针图」的 softmax
 * 和各层激活均值，这里用网页实际要跑的那份 cnn_forward.js 重算一遍对上答案。
 * 只要这里的最大误差在 1e-3 量级以内，页面上画的每一张特征图就是可信的。
 *
 * 用法： node verify_forward.js
 */

const fs = require('fs');
const path = require('path');
const F = require('./web/cnn_forward.js');

const TRACE = path.join(__dirname, 'web', 'trace_cnn.js');
const src = fs.readFileSync(TRACE, 'utf8');

const sandbox = {};
new Function('window', src + '\nreturn window.CNN_TRACE;')(sandbox);
const T = sandbox.CNN_TRACE;
if (!T) throw new Error('trace_cnn.js 里没有 CNN_TRACE，先跑 trace_cnn.py');

const A = T.arch;
const ST = F.stages(A);

/* 结构推导和实际前向的输出形状是否吻合 */
(function checkShapes() {
  const img = F.toInput(T.images[0].x, A.img);
  const act = F.forward(T.checkpoints[0].P, img, A);
  ST.forEach(function (s) {
    const a = act[s.key];
    if (!a) throw new Error('缺少层 ' + s.key);
    // flat 在 forward 里带着 (5,5,16) 的形状信息，但语义上是 400 维，单独比总数
    const shp = (s.type === 'flat') ? [a.n] : (a.data ? [a.H, a.W, a.C] : [a.length]);
    const diff = shp.length !== s.shape.length ||
      shp.some(function (v, i) { return v !== s.shape[i]; });
    if (diff) throw new Error(`层 ${s.key} 实际 ${shp} ≠ 声明 ${s.shape}`);
  });
  console.log('✓ 形状推导与前向一致：' + ST.map(s => s.shape.join('×')).join(' → '));
})();

/* 逐个快照、逐张图对答案 */
let worst = 0, worstAt = '', nChk = 0;
T.checkpoints.forEach(function (ck, ci) {
  T.images.forEach(function (im, ii) {
    const act = F.forward(ck.P, F.toInput(im.x, A.img), A);
    const ref = ck.probeProbs[ii];
    for (let k = 0; k < ref.length; k++) {
      const e = Math.abs(act.soft[k] - ref[k]);
      // NaN 比较永远为 false，会静默放过错误，所以显式拦一下
      if (!Number.isFinite(e)) throw new Error(`快照${ci} 图${ii} 类${k} 算出了非有限值`);
      if (e > worst) { worst = e; worstAt = `快照${ci} 图${ii} 类${k}`; }
    }
  });
  nChk++;
});
console.log(`✓ softmax 对账：${nChk} 个快照 × ${T.images.length} 张图，最大误差 ${worst.toExponential(2)}（${worstAt}）`);

/* 中间层也对一遍：第 0 张图的各层激活均值 */
let worstMid = 0, midAt = '';
T.checkpoints.forEach(function (ck, ci) {
  const act = F.forward(ck.P, F.toInput(T.images[0].x, A.img), A);
  Object.keys(ck.probeMeans).forEach(function (k) {
    const arr = act[k].data || act[k];
    let s = 0;
    for (let i = 0; i < arr.length; i++) s += arr[i];
    const mean = s / arr.length;
    const e = Math.abs(mean - ck.probeMeans[k]) / (Math.abs(ck.probeMeans[k]) + 1e-6);
    if (!Number.isFinite(e)) throw new Error(`快照${ci} ${k} 的均值非有限值（${mean}）`);
    if (e > worstMid) { worstMid = e; midAt = `快照${ci} ${k}`; }
  });
});
console.log(`✓ 中间层均值对账：最大相对误差 ${(worstMid * 100).toFixed(4)}%（${midAt}）`);

const ok = worst < 1e-3 && worstMid < 0.01;
console.log(ok ? '\n全部通过。' : '\n❌ 误差过大，前向实现有问题。');
process.exit(ok ? 0 : 1);
