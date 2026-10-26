/* CNN 前向计算 —— web/train.html 和 verify_forward.js 共用同一份实现。
 *
 * 目的：网页里画的每一张特征图，都必须是真的用 mnist_cnn.py 导出的权重算出来的。
 * 所以这里不重复写第二遍逻辑，页面和校验脚本都 require/引用这一个文件。
 *
 * 数据布局跟 mnist_cnn.py 完全一致：
 *   激活值  (H, W, C) 通道在最后，扁平下标 = ((y*W)+x)*C + c
 *   卷积核  W[kh][kw][ci][co]，扁平下标 = ((kh*K+kw)*Cin+ci)*Cout + co
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.CNNForward = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  /* ---------- 结构推导：把 28×28×1 一路推到 10 个 logits ---------- */
  function stages(arch) {
    var IMG = arch.img, K = arch.k, C1 = arch.c1, C2 = arch.c2,
        FC = arch.fc, NC = arch.classes;

    var h = IMG, w = IMG, c = 1;
    var rf = 1, jump = 1;              // 感受野边长、相邻激活值的输入步长
    var out = [];

    function push(o) { out.push(o); }

    push({
      key: 'input', name: '输入图像', type: 'input',
      shape: [h, w, c], count: h * w * c, rf: 1, params: 0,
      note: '28×28 的灰度图，像素除以 255 压到 0~1。网络后面所有事情都是在这个矩阵上做局部加权求和。'
    });

    // conv1
    rf += (K - 1) * jump;
    h = h - K + 1; w = w - K + 1; c = C1;
    push({
      key: 'conv1', name: 'conv 3×3 + ReLU', type: 'conv',
      shape: [h, w, c], count: h * w * c, rf: rf, params: K * K * 1 * C1 + C1,
      note: '8 个 3×3 的核在 28×28 上滑一遍，每个位置输出 8 个数。没有 padding，所以四周各掉 1 圈像素 → 26×26。ReLU 把负数砍成 0：白的地方表示"这个位置出现了我关心的某种笔画"。'
    });

    // pool1
    rf += (2 - 1) * jump; jump *= 2;
    h = Math.floor(h / 2); w = Math.floor(w / 2);
    push({
      key: 'pool1', name: 'maxpool 2×2', type: 'pool',
      shape: [h, w, c], count: h * w * c, rf: rf, params: 0,
      note: '每 2×2 窗口只留最大值。尺寸减半、计算量降到 1/4，同时让后面一层"看到"的范围变大（平移一点也不会变）。注意特征图变得有点方块感，这是池化的代价。'
    });

    // conv2
    rf += (K - 1) * jump;
    h = h - K + 1; w = w - K + 1; c = C2;
    push({
      key: 'conv2', name: 'conv 3×3 + ReLU', type: 'conv',
      shape: [h, w, c], count: h * w * c, rf: rf, params: K * K * C1 * C2 + C2,
      note: '16 个核，但每个核要看 8 个输入通道再求和，所以这一层参数最多、也最"抽象"——它组合的是上一层的笔画片段，不是原始像素。11×11 是奇数，下一个池化会先裁掉一行一列。'
    });

    // pool2
    rf += (2 - 1) * jump; jump *= 2;
    h = Math.floor(h / 2); w = Math.floor(w / 2);
    push({
      key: 'pool2', name: 'maxpool 2×2', type: 'pool',
      shape: [h, w, c], count: h * w * c, rf: rf, params: 0,
      note: '5×5×16 是整个 CNN 的"总结"：16 个通道各自把图缩成 5×5 的粗略分布。到这一步已经看不出数字长什么样了，只剩下"哪里有什么结构"。'
    });

    // flatten
    var flat = h * w * c;
    push({
      key: 'flat', name: 'flatten', type: 'flat',
      shape: [flat], count: flat, rf: rf, params: 0,
      note: '把 5×5×16 摊成 400 个数。展开顺序是通道在最后（channel-last）：前 25 个数是通道 0 的 5×5，接着 25 个是通道 1……所以下面这张 20×20 的方块里，每一行 25 格就是一个通道。'
    });

    // fc1
    push({
      key: 'fc1', name: 'fc ' + flat + '→' + FC + ' + ReLU', type: 'fc',
      shape: [FC], count: FC, rf: rf, params: flat * FC + FC,
      note: '全连接：每个隐藏单元看全部 400 个输入，学的是"这些结构同时出现时意味着什么"。这一层参数 25,664 个，占了整个模型的 93%。'
    });

    // fc2
    push({
      key: 'fc2', name: 'fc ' + FC + '→' + NC, type: 'fc',
      shape: [NC], count: NC, rf: rf, params: FC * NC + NC,
      note: '10 个 logits，没有 softmax 的原分值。有正有负，只比较大小，所以不能当概率看。'
    });

    push({
      key: 'soft', name: 'softmax', type: 'soft',
      shape: [NC], count: NC, rf: rf, params: 0,
      note: '减去最大值再取 exp 归一化，变成 10 个加起来等于 1 的概率。截图里"置信度"就是这个。'
    });

    return out;
  }

  /* ---------- 算子 ---------- */
  // x: Float32Array (H*W*Cin) 通道在最后；Wf: 扁平 [kh][kw][ci][co]
  function conv3x3(x, H, W, Cin, Cout, Wf, bf, K) {
    var oh = H - K + 1, ow = W - K + 1;
    var out = new Float32Array(oh * ow * Cout);
    var planeW = Cin * Cout;
    for (var y = 0; y < oh; y++) {
      for (var xx = 0; xx < ow; xx++) {
        var o = (y * ow + xx) * Cout;
        for (var c0 = 0; c0 < Cout; c0++) out[o + c0] = bf[c0];
        for (var kh = 0; kh < K; kh++) {
          for (var kw = 0; kw < K; kw++) {
            var src = ((y + kh) * W + (xx + kw)) * Cin;
            var wOff = (kh * K + kw) * planeW;
            for (var ci = 0; ci < Cin; ci++) {
              var v = x[src + ci];
              if (v === 0) continue;
              var wb = wOff + ci * Cout;
              for (var co = 0; co < Cout; co++) out[o + co] += v * Wf[wb + co];
            }
          }
        }
        for (var c1 = 0; c1 < Cout; c1++) if (out[o + c1] < 0) out[o + c1] = 0;  // ReLU
      }
    }
    return { H: oh, W: ow, C: Cout, data: out };
  }

  function maxpool2(m) {
    var H = m.H, W = m.W, C = m.C;
    var h = H >> 1, w = W >> 1;       // 奇数边长时裁掉最后一行/列，与 numpy 版一致
    var out = new Float32Array(h * w * C);
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        for (var c = 0; c < C; c++) {
          var mx = -Infinity;
          for (var dy = 0; dy < 2; dy++) {
            for (var dx = 0; dx < 2; dx++) {
              var v = m.data[((y * 2 + dy) * W + (x * 2 + dx)) * C + c];
              if (v > mx) mx = v;
            }
          }
          out[(y * w + x) * C + c] = mx;
        }
      }
    }
    return { H: h, W: w, C: C, data: out };
  }

  /* 权重只做一次 Float32 转换，之后每个快照复用（切图、切快照都会重算前向） */
  var wcache = new WeakMap();
  function typed(P) {
    var t = wcache.get(P);
    if (t) return t;
    t = {
      W1: new Float32Array(P.W1), b1: new Float32Array(P.b1),
      W2: new Float32Array(P.W2), b2: new Float32Array(P.b2),
      Wf: new Float32Array(P.Wf), bf: new Float32Array(P.bf),
      Wo: new Float32Array(P.Wo), bo: new Float32Array(P.bo)
    };
    wcache.set(P, t);
    return t;
  }

  /* input: Float32Array(28*28)，已是 0~1；P 是 checkpoints[i].P（参数对象） */
  function forward(P, input, arch) {
    if (!P || !P.W1) throw new Error('forward 需要参数对象（checkpoints[i].P），传进来的是 ' + typeof P);
    var A = arch, w = typed(P);
    var img = { H: A.img, W: A.img, C: 1, data: input };

    var conv1 = conv3x3(input, A.img, A.img, 1, A.c1, w.W1, w.b1, A.k);
    var pool1 = maxpool2(conv1);
    var conv2 = conv3x3(pool1.data, pool1.H, pool1.W, A.c1, A.c2, w.W2, w.b2, A.k);
    var pool2 = maxpool2(conv2);

    var F = pool2.H * pool2.W * pool2.C;
    var fc1 = new Float32Array(A.fc);
    for (var j = 0; j < A.fc; j++) {
      var s = w.bf[j];
      for (var i = 0; i < F; i++) {
        var v = pool2.data[i];
        if (v !== 0) s += v * w.Wf[i * A.fc + j];
      }
      fc1[j] = s > 0 ? s : 0;
    }

    var logits = new Float32Array(A.classes), mx = -Infinity;
    for (var k = 0; k < A.classes; k++) {
      var t = w.bo[k];
      for (var j2 = 0; j2 < A.fc; j2++) t += fc1[j2] * w.Wo[j2 * A.classes + k];
      logits[k] = t;
      if (t > mx) mx = t;
    }

    var probs = new Float32Array(A.classes), sum = 0;
    for (var k2 = 0; k2 < A.classes; k2++) { probs[k2] = Math.exp(logits[k2] - mx); sum += probs[k2]; }
    for (var k3 = 0; k3 < A.classes; k3++) probs[k3] /= sum;

    return {
      input: img, conv1: conv1, pool1: pool1, conv2: conv2, pool2: pool2,
      flat: { H: pool2.H, W: pool2.W, C: pool2.C, data: pool2.data, n: F },
      fc1: fc1, fc2: logits, soft: probs
    };
  }

  /* 图片数组（0~255 的整数）→ 模型输入（0~1） */
  function toInput(json, n) {
    var out = new Float32Array(n * n);
    for (var i = 0; i < n * n; i++) out[i] = json[i] / 255;
    return out;
  }

  return { stages: stages, forward: forward, toInput: toInput };
});
