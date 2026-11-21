/* sketch.js — 手绘原语: 一门小型绘图语言 (靛蓝手绘族)
 *
 * 用法:
 *   Sketch.installDefs();                       // 幂等注入公共 defs (rough 滤镜 + 箭头)
 *   const s = Sketch(svgEl);                    // 取一个 <svg> 作为画布
 *   s.path(d, {stroke, width})                  // 手绘路径 (自动挂 rough 抖动)
 *   s.node(x, y, {r})                           // 手绘圆节点
 *   s.label(x, y, text, {size, color})          // 手写标注 (拉丁 Caveat / 中文楷体栈)
 *   s.dot(x, y, color, r)                       // 静态点
 *   s.flow(d, {color, dur, delay})              // 沿路径循环流动的光点
 *   s.pulse(d, {t0, travel, period})            // 全局时间轴脉冲 (第 t0 秒出发, travel 秒走完)
 *   s.path(d, {fill}) / s.label(..., {mono})    // 色块填充 / 等宽标签
 *
 * 组合示例见 docs-book/sketchbook.html
 */
const Sketch = (() => {
  const NS = 'http://www.w3.org/2000/svg';
  const HAND = 'Caveat, "Xingkai SC", "Kaiti SC", KaiTi, cursive';
  const PALETTE = { ink: '#3a4aa8', deep: '#2c3a8c', amber: '#bf8700', green: '#1a7f37', dim: '#9aa1b9' };
  let uid = 0;

  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  function installDefs() {
    if (document.getElementById('sk-defs')) return;
    const holder = el('svg', { width: 0, height: 0, style: 'position:absolute' }, document.body);
    holder.id = 'sk-defs';
    const defs = el('defs', {}, holder);
    const f = el('filter', { id: 'sk-rough', x: '-5%', y: '-5%', width: '110%', height: '110%' }, defs);
    el('feTurbulence', { type: 'fractalNoise', baseFrequency: '0.03', numOctaves: '2', seed: '7' }, f);
    el('feDisplacementMap', { in: 'SourceGraphic', scale: '3.2' }, f);
    const mk = el('marker', { id: 'sk-arrow', viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 9, markerHeight: 9, orient: 'auto-start-reverse' }, defs);
    el('path', { d: 'M1 1 L9 5 L1 9', fill: 'none', stroke: PALETTE.ink, 'stroke-width': 1.4, 'stroke-linecap': 'round' }, mk);
  }

  return function sketch(svg) {
    installDefs();
    const rough = el('g', { filter: 'url(#sk-rough)' }, svg);   // 手绘层: 线条抖动
    const plain = el('g', {}, svg);                              // 清晰层: 文字/光点

    return {
      svg, PALETTE,
      path(d, { stroke = PALETTE.ink, width = 1.6, arrow = false, opacity = 1, fill = 'none', rough: useRough = true } = {}) {
        return el('path', { d, stroke, 'stroke-width': width, fill, 'stroke-linecap': 'round',
                            opacity, ...(arrow ? { 'marker-end': 'url(#sk-arrow)' } : {}) },
                  useRough ? rough : plain);
      },
      node(x, y, { r = 5.5, fill = '#fff', stroke = PALETTE.ink } = {}) {
        return el('circle', { cx: x, cy: y, r, fill, stroke, 'stroke-width': 1.6 }, rough);
      },
      label(x, y, text, { size = 19, color = PALETTE.dim, anchor = 'middle', weight = 500, mono = false } = {}) {
        const family = mono ? 'ui-monospace, Menlo, monospace' : HAND;
        const t = el('text', { x, y, 'text-anchor': anchor, fill: color,
                               style: `font: ${weight} ${size}px ${family}` }, plain);
        t.textContent = text;
        return t;
      },
      dot(x, y, color = PALETTE.amber, r = 4) {
        return el('circle', { cx: x, cy: y, r, fill: color }, plain);
      },
      flow(d, { color = PALETTE.amber, size = 4, dur = 3.2, delay = 0, from = 0 } = {}) {
        const id = 'sk-path-' + (++uid);
        el('path', { id, d, stroke: 'none', fill: 'none' }, plain);
        const c = el('circle', { r: size, fill: color, opacity: 0 }, plain);
        const am = el('animateMotion', { dur: dur + 's', repeatCount: 'indefinite', begin: delay + 's' }, c);
        el('mpath', { href: '#' + id }, am);
        el('animate', { attributeName: 'opacity', from: 0, to: 1, dur: '0.3s', begin: delay + 's', fill: 'freeze' }, c);
        el('animate', { attributeName: 'opacity', from: 1, to: 0, dur: '0.3s',
                        begin: (delay + dur - 0.3) + 's', repeatCount: 'indefinite' }, c);
        return c;
      },
      pulse(d, { t0 = 0, travel = 0.8, period = 10, color = PALETTE.amber, size = 4 } = {}) {
        const id = 'sk-p-' + (++uid);
        el('path', { id, d, fill: 'none', stroke: 'none' }, plain);
        const c = el('circle', { r: size, fill: color, opacity: 0 }, plain);
        const frac = travel / period;
        const am = el('animateMotion', { dur: period + 's', begin: t0 + 's', repeatCount: 'indefinite',
                                         keyPoints: '0;1;1', keyTimes: '0;' + frac + ';1', calcMode: 'linear' }, c);
        el('mpath', { href: '#' + id }, am);
        el('animate', { attributeName: 'opacity', values: '0;1;1;0;0',
                        keyTimes: '0;0.015;' + Math.max(frac - 0.015, 0.03) + ';' + frac + ';1',
                        dur: period + 's', begin: t0 + 's', repeatCount: 'indefinite' }, c);
        return c;
      },
    };
  };
})();
