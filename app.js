/**
 * app.js - Educational Perceptron Learning Simulator
 * High School Computer Science (AI Fundamentals)
 * Pure Vanilla JS + Canvas + Chart.js
 */

(function () {
  'use strict';

  // ==========================================================================
  // 1. Application State
  // ==========================================================================
  let points = [];           // Array of { id, x, y, label }
  let w1 = 0.0;             // Weight for x
  let w2 = 0.0;             // Weight for y
  let b = 0.0;              // Bias
  let learningRate = 0.1;   // Eta
  let epoch = 0;            // Epoch / Iteration count
  let stepCount = 0;        // Total weight updates
  let isPlaying = false;    // Auto play state
  let playInterval = null;  // Interval timer reference
  let speedMs = 300;        // Auto play speed in ms

  let mode = 'blue';        // 'blue' (+1), 'red' (-1), 'select' (drag/delete)
  let selectedPointId = null;
  let isDragging = false;
  
  let chartViewMode = 'loss'; // 'loss' or 'weights'
  let history = [];          // Array of { step, epoch, errors, w1, w2, b }
  let lastStepLog = null;    // Detailed info of last weight update

  // Math Domain Bounds [-10, 10]
  const DOMAIN = 10;
  const CANVAS_MARGIN = 35; // Pixels reserved for grid label margins

  // DOM Element References
  const canvas = document.getElementById('coordCanvas');
  const ctx = canvas.getContext('2d');
  
  // Dashboard Stat Elements
  const statW1 = document.getElementById('statW1');
  const statW2 = document.getElementById('statW2');
  const statB = document.getElementById('statB');
  const statEpoch = document.getElementById('statEpoch');
  const statError = document.getElementById('statError');
  const statTotalPoints = document.getElementById('statTotalPoints');
  const statStatus = document.getElementById('statStatus');

  // Control Inputs & Buttons
  const modeBlueBtn = document.getElementById('modeBlueBtn');
  const modeRedBtn = document.getElementById('modeRedBtn');
  const modeSelectBtn = document.getElementById('modeSelectBtn');
  const presetSelect = document.getElementById('presetSelect');
  
  const learningRateInput = document.getElementById('learningRateInput');
  const lrValDisplay = document.getElementById('lrValDisplay');
  const speedInput = document.getElementById('speedInput');
  const speedValDisplay = document.getElementById('speedValDisplay');

  const autoPlayBtn = document.getElementById('autoPlayBtn');
  const autoPlayBtnText = document.getElementById('autoPlayBtnText');
  const stepBtn = document.getElementById('stepBtn');
  const resetWeightsBtn = document.getElementById('resetWeightsBtn');
  const clearPointsBtn = document.getElementById('clearPointsBtn');

  const coordDisplay = document.getElementById('coordDisplay');
  const equationText = document.getElementById('equationText');
  const canvasEmptyHint = document.getElementById('canvasEmptyHint');

  const chartTabLoss = document.getElementById('chartTabLoss');
  const chartTabWeights = document.getElementById('chartTabWeights');
  const stepDetailBox = document.getElementById('stepDetailBox');

  const guideBtn = document.getElementById('guideBtn');
  const guideModal = document.getElementById('guideModal');
  const closeGuideBtn = document.getElementById('closeGuideBtn');
  const confirmGuideBtn = document.getElementById('confirmGuideBtn');

  let lossChart = null;

  // ==========================================================================
  // 2. Coordinate Conversion Utilities
  // ==========================================================================
  function getCanvasSize() {
    const rect = canvas.getBoundingClientRect();
    return {
      width: rect.width || 400,
      height: rect.height || 400
    };
  }

  function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const size = Math.min(rect.width, rect.height) || 400;

    canvas.width = size * dpr;
    canvas.height = size * dpr;
    ctx.scale(dpr, dpr);
    
    renderCanvas();
  }

  function mathToCanvas(x, y) {
    const rect = canvas.getBoundingClientRect();
    const size = rect.width || 400;
    const plotSize = size - 2 * CANVAS_MARGIN;

    const cx = CANVAS_MARGIN + ((x + DOMAIN) / (2 * DOMAIN)) * plotSize;
    const cy = CANVAS_MARGIN + ((DOMAIN - y) / (2 * DOMAIN)) * plotSize;
    return { cx, cy };
  }

  function canvasToMath(cx, cy) {
    const rect = canvas.getBoundingClientRect();
    const size = rect.width || 400;
    const plotSize = size - 2 * CANVAS_MARGIN;

    let x = ((cx - CANVAS_MARGIN) / plotSize) * (2 * DOMAIN) - DOMAIN;
    let y = DOMAIN - ((cy - CANVAS_MARGIN) / plotSize) * (2 * DOMAIN);

    // Clamp to domain
    x = Math.max(-DOMAIN, Math.min(DOMAIN, x));
    y = Math.max(-DOMAIN, Math.min(DOMAIN, y));

    // Round to 1 decimal place
    return {
      x: Math.round(x * 10) / 10,
      y: Math.round(y * 10) / 10
    };
  }


  // ==========================================================================
  // 3. Perceptron Core Algorithm
  // ==========================================================================
  function predict(x, y, w1Val = w1, w2Val = w2, bVal = b) {
    const z = w1Val * x + w2Val * y + bVal;
    return z >= 0 ? 1 : -1;
  }

  function countMisclassifications() {
    let errors = 0;
    for (const p of points) {
      if (predict(p.x, p.y) !== p.label) {
        errors++;
      }
    }
    return errors;
  }

  function getMisclassifiedPoints() {
    return points.filter(p => predict(p.x, p.y) !== p.label);
  }

  /**
   * Executes 1 update step on a single misclassified point.
   * Perceptron Learning Rule:
   *   error = target - prediction  (where target in {+1, -1})
   *   w1 <- w1 + eta * target * x
   *   w2 <- w2 + eta * target * y
   *   b  <- b  + eta * target
   */
  function trainSingleStep() {
    if (points.length === 0) {
      updateStatus('points_needed', '점이 없습니다. 점을 먼저 찍어보세요.');
      stopAutoPlay();
      return false;
    }

    const misclassified = getMisclassifiedPoints();

    if (misclassified.length === 0) {
      updateStatus('success', '🟢 모든 점이 완벽히 분류되었습니다! (학습 완료)');
      stopAutoPlay();
      return true;
    }

    // Pick the first misclassified point to fix
    const targetPoint = misclassified[0];
    const oldW1 = w1;
    const oldW2 = w2;
    const oldB = b;

    const x = targetPoint.x;
    const y = targetPoint.y;
    const t = targetPoint.label; // +1 or -1
    const pred = predict(x, y);

    // Weight Update Calculation
    const deltaW1 = learningRate * t * x;
    const deltaW2 = learningRate * t * y;
    const deltaB = learningRate * t;

    w1 += deltaW1;
    w2 += deltaW2;
    b += deltaB;

    stepCount++;
    epoch++;

    // Save update log for inspector
    lastStepLog = {
      point: targetPoint,
      oldW1, oldW2, oldB,
      newW1: w1, newW2: w2, newB: b,
      deltaW1, deltaW2, deltaB,
      t, pred,
      x, y,
      eta: learningRate
    };

    // Record history for chart
    const currentErrors = countMisclassifications();
    history.push({
      step: stepCount,
      epoch: epoch,
      errors: currentErrors,
      w1: Number(w1.toFixed(3)),
      w2: Number(w2.toFixed(3)),
      b: Number(b.toFixed(3))
    });

    // Refresh UI
    updateStats();
    updateStepLogUI();
    updateChart();
    renderCanvas();

    if (currentErrors === 0) {
      updateStatus('success', '🟢 모든 점이 완벽히 분류되었습니다!');
      stopAutoPlay();
      return true;
    } else {
      updateStatus('training', `🟡 가중치 수정 중... (남은 오분류: ${currentErrors}개)`);
    }

    return false;
  }

  function resetWeights() {
    w1 = 0.0;
    w2 = 0.0;
    b = 0.0;
    epoch = 0;
    stepCount = 0;
    history = [];
    lastStepLog = null;

    if (points.length > 0) {
      const errs = countMisclassifications();
      history.push({ step: 0, epoch: 0, errors: errs, w1: 0, w2: 0, b: 0 });
    }

    stopAutoPlay();
    updateStats();
    updateStepLogUI();
    updateChart();
    renderCanvas();
    updateStatus('idle', '⚪ 가중치가 초기화되었습니다. 학습을 시작해보세요.');
  }


  // ==========================================================================
  // 4. Auto Play Controls
  // ==========================================================================
  function startAutoPlay() {
    if (points.length === 0) {
      alert('좌표평면에 점을 1개 이상 배치한 후 학습을 시작해 주세요.');
      return;
    }

    if (countMisclassifications() === 0) {
      alert('현재 모든 점이 이미 정확히 분류되어 있습니다! 가중치 초기화 후 다시 시도해 보세요.');
      return;
    }

    isPlaying = true;
    autoPlayBtn.classList.remove('bg-indigo-600', 'hover:bg-indigo-700');
    autoPlayBtn.classList.add('bg-amber-600', 'hover:bg-amber-700');
    autoPlayBtnText.textContent = '일시 정지';
    autoPlayBtn.querySelector('i').className = 'fa-solid fa-pause';

    playInterval = setInterval(() => {
      const finished = trainSingleStep();
      if (finished || epoch >= 200) {
        if (epoch >= 200 && countMisclassifications() > 0) {
          updateStatus('failed', '⚠️ 200회 반복 후에도 분류에 실패했습니다. (XOR 문제 가능성)');
        }
        stopAutoPlay();
      }
    }, speedMs);
  }

  function stopAutoPlay() {
    isPlaying = false;
    if (playInterval) {
      clearInterval(playInterval);
      playInterval = null;
    }
    autoPlayBtn.classList.remove('bg-amber-600', 'hover:bg-amber-700');
    autoPlayBtn.classList.add('bg-indigo-600', 'hover:bg-indigo-700');
    autoPlayBtnText.textContent = '자동 학습 시작';
    autoPlayBtn.querySelector('i').className = 'fa-solid fa-play';
  }


  // ==========================================================================
  // 5. Presets Generator
  // ==========================================================================
  function loadPreset(presetName) {
    stopAutoPlay();
    points = [];

    switch (presetName) {
      case 'and':
        // AND Gate: (1,1)->+1, others -> -1
        points = [
          { id: 'p1', x: 5, y: 5, label: 1 },
          { id: 'p2', x: -5, y: 5, label: -1 },
          { id: 'p3', x: 5, y: -5, label: -1 },
          { id: 'p4', x: -5, y: -5, label: -1 }
        ];
        break;

      case 'or':
        // OR Gate: (-1,-1)->-1, others -> +1
        points = [
          { id: 'p1', x: 5, y: 5, label: 1 },
          { id: 'p2', x: -5, y: 5, label: 1 },
          { id: 'p3', x: 5, y: -5, label: 1 },
          { id: 'p4', x: -5, y: -5, label: -1 }
        ];
        break;

      case 'xor':
        // XOR Gate: (1,1), (-1,-1)->-1; (-1,1), (1,-1)->+1 (Non-linearly separable!)
        points = [
          { id: 'p1', x: 5, y: 5, label: -1 },
          { id: 'p2', x: -5, y: -5, label: -1 },
          { id: 'p3', x: -5, y: 5, label: 1 },
          { id: 'p4', x: 5, y: -5, label: 1 }
        ];
        break;

      case 'diagonal':
        // Linearly separable along diagonal y = -x + 1
        points = [
          { id: 'p1', x: -6, y: 7, label: 1 },
          { id: 'p2', x: -2, y: 8, label: 1 },
          { id: 'p3', x: 2, y: 5, label: 1 },
          { id: 'p4', x: -8, y: 2, label: 1 },
          { id: 'p5', x: -3, y: 3, label: 1 },
          { id: 'p6', x: 4, y: -3, label: -1 },
          { id: 'p7', x: 8, y: -2, label: -1 },
          { id: 'p8', x: 3, y: -7, label: -1 },
          { id: 'p9', x: 7, y: -6, label: -1 },
          { id: 'p10', x: -2, y: -5, label: -1 }
        ];
        break;

      case 'random':
        for (let i = 0; i < 10; i++) {
          const rx = Math.round((Math.random() * 14 - 7) * 10) / 10;
          const ry = Math.round((Math.random() * 14 - 7) * 10) / 10;
          const label = (rx + ry > 0) ? 1 : -1;
          points.push({
            id: 'rand_' + i + '_' + Date.now(),
            x: rx,
            y: ry,
            label: label
          });
        }
        break;

      case 'custom':
      default:
        points = [];
        break;
    }

    resetWeights();
  }


  // ==========================================================================
  // 6. Canvas Rendering
  // ==========================================================================
  function renderCanvas() {
    const rect = canvas.getBoundingClientRect();
    const size = rect.width || 400;

    // Clear canvas
    ctx.clearRect(0, 0, size, size);

    // 1. Draw Shaded Decision Regions (Blue vs Red Zone)
    if (w1 !== 0 || w2 !== 0 || b !== 0) {
      drawDecisionRegions(size);
    }

    // 2. Draw Grid & Axes
    drawGridAndAxes(size);

    // 3. Draw Decision Boundary Line (w1*x + w2*y + b = 0)
    if (w1 !== 0 || w2 !== 0 || b !== 0) {
      drawBoundaryLine(size);
    }

    // 4. Draw Data Points
    drawPoints(size);

    // 5. Empty Canvas Hint Visibility
    if (points.length === 0) {
      canvasEmptyHint.classList.remove('opacity-0', 'pointer-events-none');
    } else {
      canvasEmptyHint.classList.add('opacity-0', 'pointer-events-none');
    }
  }

  function drawGridAndAxes(size) {
    const plotSize = size - 2 * CANVAS_MARGIN;
    const step = plotSize / (2 * DOMAIN);

    ctx.save();

    // Light Grid Lines
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1;
    ctx.setFont = '10px monospace';
    ctx.fillStyle = '#94a3b8';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    for (let val = -DOMAIN; val <= DOMAIN; val += 2) {
      const { cx, cy } = mathToCanvas(val, val);

      // Vertical grid line
      ctx.beginPath();
      ctx.moveTo(cx, CANVAS_MARGIN);
      ctx.lineTo(cx, size - CANVAS_MARGIN);
      ctx.stroke();

      // Horizontal grid line
      ctx.beginPath();
      ctx.moveTo(CANVAS_MARGIN, cy);
      ctx.lineTo(size - CANVAS_MARGIN, cy);
      ctx.stroke();

      // Axis Numbers
      if (val !== 0) {
        // X-axis label
        ctx.fillText(val.toString(), cx, size - CANVAS_MARGIN + 14);
        // Y-axis label
        ctx.fillText(val.toString(), CANVAS_MARGIN - 14, cy);
      }
    }

    // Main Axes (X and Y)
    const origin = mathToCanvas(0, 0);

    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 2;

    // X-Axis
    ctx.beginPath();
    ctx.moveTo(CANVAS_MARGIN - 5, origin.cy);
    ctx.lineTo(size - CANVAS_MARGIN + 10, origin.cy);
    ctx.stroke();

    // Y-Axis
    ctx.beginPath();
    ctx.moveTo(origin.cx, size - CANVAS_MARGIN + 5);
    ctx.lineTo(origin.cx, CANVAS_MARGIN - 10);
    ctx.stroke();

    // Arrows on X and Y
    ctx.fillStyle = '#334155';
    // X arrow
    ctx.beginPath();
    ctx.moveTo(size - CANVAS_MARGIN + 12, origin.cy);
    ctx.lineTo(size - CANVAS_MARGIN + 5, origin.cy - 4);
    ctx.lineTo(size - CANVAS_MARGIN + 5, origin.cy + 4);
    ctx.fill();

    // Y arrow
    ctx.beginPath();
    ctx.moveTo(origin.cx, CANVAS_MARGIN - 12);
    ctx.lineTo(origin.cx - 4, CANVAS_MARGIN - 5);
    ctx.lineTo(origin.cx + 4, CANVAS_MARGIN - 5);
    ctx.fill();

    // Labels X, Y, O
    ctx.font = 'bold 12px Pretendard, sans-serif';
    ctx.fillText('X', size - CANVAS_MARGIN + 18, origin.cy + 2);
    ctx.fillText('Y', origin.cx, CANVAS_MARGIN - 20);
    ctx.fillText('O', origin.cx - 10, origin.cy + 10);

    ctx.restore();
  }

  function drawDecisionRegions(size) {
    const resolution = 8; // Pixel grid step for shading
    ctx.save();

    for (let px = CANVAS_MARGIN; px <= size - CANVAS_MARGIN; px += resolution) {
      for (let py = CANVAS_MARGIN; py <= size - CANVAS_MARGIN; py += resolution) {
        const mathPos = canvasToMath(px + resolution / 2, py + resolution / 2);
        const z = w1 * mathPos.x + w2 * mathPos.y + b;

        if (z >= 0) {
          ctx.fillStyle = 'rgba(59, 130, 246, 0.08)'; // Light Blue
        } else {
          ctx.fillStyle = 'rgba(239, 68, 68, 0.08)';  // Light Red
        }

        ctx.fillRect(px, py, resolution, resolution);
      }
    }

    ctx.restore();
  }

  function drawBoundaryLine(size) {
    ctx.save();

    // Find intersection points of line w1*x + w2*y + b = 0 with bounds [-10, 10]
    const bounds = [-DOMAIN, DOMAIN];
    let linePoints = [];

    if (Math.abs(w2) > 0.0001) {
      // Calculate y for x = -10 and x = 10
      for (const xVal of bounds) {
        const yVal = (-w1 * xVal - b) / w2;
        if (yVal >= -DOMAIN && yVal <= DOMAIN) {
          linePoints.push({ x: xVal, y: yVal });
        }
      }
    }

    if (Math.abs(w1) > 0.0001) {
      // Calculate x for y = -10 and y = 10
      for (const yVal of bounds) {
        const xVal = (-w2 * yVal - b) / w1;
        if (xVal >= -DOMAIN && xVal <= DOMAIN) {
          // Avoid duplicate points
          if (!linePoints.some(p => Math.abs(p.x - xVal) < 0.001 && Math.abs(p.y - yVal) < 0.001)) {
            linePoints.push({ x: xVal, y: yVal });
          }
        }
      }
    }

    if (linePoints.length >= 2) {
      const p1 = mathToCanvas(linePoints[0].x, linePoints[0].y);
      const p2 = mathToCanvas(linePoints[1].x, linePoints[1].y);

      // Draw Shadow Line
      ctx.shadowColor = 'rgba(79, 70, 229, 0.4)';
      ctx.shadowBlur = 8;
      ctx.strokeStyle = '#4f46e5'; // Deep Indigo
      ctx.lineWidth = 3.5;

      ctx.beginPath();
      ctx.moveTo(p1.cx, p1.cy);
      ctx.lineTo(p2.cx, p2.cy);
      ctx.stroke();

      ctx.shadowBlur = 0; // Reset shadow

      // Draw Normal Vector Arrow W = (w1, w2) to show positive direction (+1 side)
      const midMathX = (linePoints[0].x + linePoints[1].x) / 2;
      const midMathY = (linePoints[0].y + linePoints[1].y) / 2;
      const midCanvas = mathToCanvas(midMathX, midMathY);

      // Scale weight vector for visual arrow length
      const normLen = Math.sqrt(w1 * w1 + w2 * w2) || 1;
      const arrowMathX = midMathX + (w1 / normLen) * 2.5;
      const arrowMathY = midMathY + (w2 / normLen) * 2.5;
      const arrowCanvas = mathToCanvas(arrowMathX, arrowMathY);

      ctx.strokeStyle = '#2563eb';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(midCanvas.cx, midCanvas.cy);
      ctx.lineTo(arrowCanvas.cx, arrowCanvas.cy);
      ctx.stroke();

      // Arrow head
      const angle = Math.atan2(arrowCanvas.cy - midCanvas.cy, arrowCanvas.cx - midCanvas.cx);
      ctx.fillStyle = '#2563eb';
      ctx.beginPath();
      ctx.moveTo(arrowCanvas.cx, arrowCanvas.cy);
      ctx.lineTo(arrowCanvas.cx - 8 * Math.cos(angle - Math.PI / 6), arrowCanvas.cy - 8 * Math.sin(angle - Math.PI / 6));
      ctx.lineTo(arrowCanvas.cx - 8 * Math.cos(angle + Math.PI / 6), arrowCanvas.cy - 8 * Math.sin(angle + Math.PI / 6));
      ctx.fill();

      // Text (+1 영역)
      ctx.fillStyle = '#1e40af';
      ctx.font = 'bold 11px Pretendard, sans-serif';
      ctx.fillText('+1 영역', arrowCanvas.cx + 8, arrowCanvas.cy + 4);
    }

    ctx.restore();
  }

  function drawPoints(size) {
    const radius = 8;

    for (const p of points) {
      const { cx, cy } = mathToCanvas(p.x, p.y);
      const isCorrect = predict(p.x, p.y) === p.label;
      const isSelected = p.id === selectedPointId;

      ctx.save();

      // Highlight Misclassified Points with glowing ring
      if (!isCorrect) {
        ctx.strokeStyle = '#ef4444'; // Red glowing warning
        ctx.lineWidth = 3;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.arc(cx, cy, radius + 6, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Highlight Selected Point
      if (isSelected) {
        ctx.strokeStyle = '#f59e0b'; // Amber selection ring
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.arc(cx, cy, radius + 4, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Circle Fill
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);

      if (p.label === 1) {
        ctx.fillStyle = '#2563eb'; // Blue
        ctx.strokeStyle = '#ffffff';
      } else {
        ctx.fillStyle = '#dc2626'; // Red
        ctx.strokeStyle = '#ffffff';
      }

      ctx.lineWidth = 2;
      ctx.fill();
      ctx.stroke();

      // Label (+ or -) inside point
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 11px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(p.label === 1 ? '+' : '-', cx, cy);

      ctx.restore();
    }
  }


  // ==========================================================================
  // 7. UI Stats & Inspector Log Updates
  // ==========================================================================
  function updateStats() {
    statW1.textContent = w1.toFixed(2);
    statW2.textContent = w2.toFixed(2);
    statB.textContent = b.toFixed(2);
    statEpoch.textContent = epoch;

    const errors = countMisclassifications();
    statError.textContent = errors;
    statTotalPoints.textContent = `/ ${points.length}개`;

    // Equation display text
    const signW2 = w2 >= 0 ? '+' : '-';
    const signB = b >= 0 ? '+' : '-';
    equationText.textContent = `${w1.toFixed(2)}x ${signW2} ${Math.abs(w2).toFixed(2)}y ${signB} ${Math.abs(b).toFixed(2)} = 0`;
  }

  function updateStatus(type, message) {
    if (type === 'success') {
      statStatus.className = 'text-xs font-bold text-emerald-600 mt-1 flex items-center gap-1.5';
      statStatus.innerHTML = `<span class="w-2.5 h-2.5 rounded-full bg-emerald-500"></span> ${message}`;
    } else if (type === 'training') {
      statStatus.className = 'text-xs font-bold text-amber-600 mt-1 flex items-center gap-1.5';
      statStatus.innerHTML = `<span class="w-2.5 h-2.5 rounded-full bg-amber-500 animate-pulse"></span> ${message}`;
    } else if (type === 'failed') {
      statStatus.className = 'text-xs font-bold text-rose-600 mt-1 flex items-center gap-1.5';
      statStatus.innerHTML = `<span class="w-2.5 h-2.5 rounded-full bg-rose-500"></span> ${message}`;
    } else {
      statStatus.className = 'text-xs font-bold text-slate-600 mt-1 flex items-center gap-1.5';
      statStatus.innerHTML = `<span class="w-2.5 h-2.5 rounded-full bg-slate-400"></span> ${message}`;
    }
  }

  function updateStepLogUI() {
    if (!lastStepLog) {
      stepDetailBox.innerHTML = `
        <div class="text-slate-400 text-center py-4">
          <i class="fa-solid fa-arrows-rotate text-2xl mb-1 text-slate-300 block"></i>
          학습 버튼을 누르면 이 곳에 구체적인 가중치 계산 과정이 표시됩니다.
        </div>
      `;
      return;
    }

    const log = lastStepLog;
    const targetColorClass = log.t === 1 ? 'text-blue-600 font-bold' : 'text-rose-600 font-bold';
    const targetText = log.t === 1 ? '파란 점 (+1)' : '빨간 점 (-1)';
    const predText = log.pred === 1 ? '파란 영역 (+1)' : '빨간 영역 (-1)';

    stepDetailBox.innerHTML = `
      <div class="space-y-2">
        <div class="flex items-center justify-between font-semibold pb-1.5 border-b border-slate-200">
          <span class="text-slate-700">📍 수정 대상 점 P(${log.x}, ${log.y})</span>
          <span class="text-[11px] px-2 py-0.5 rounded bg-slate-200 text-slate-700">Step ${stepCount}</span>
        </div>

        <div class="grid grid-cols-2 gap-2 text-[11px] bg-white p-2 rounded-lg border border-slate-200">
          <div>실제 정답: <span class="${targetColorClass}">${targetText}</span></div>
          <div>퍼셉트론 예측: <span class="font-bold text-slate-700">${predText}</span></div>
        </div>

        <div class="text-[11px] font-mono space-y-1 bg-slate-900 text-slate-100 p-2.5 rounded-lg">
          <div class="text-slate-400 text-[10px] mb-1">▼ 가중치 변경 내역 (학습률 η = ${log.eta})</div>
          <div>w1: ${log.oldW1.toFixed(2)} → <span class="text-amber-300 font-bold">${log.newW1.toFixed(2)}</span> (${log.deltaW1 >= 0 ? '+' : ''}${log.deltaW1.toFixed(2)})</div>
          <div>w2: ${log.oldW2.toFixed(2)} → <span class="text-blue-300 font-bold">${log.newW2.toFixed(2)}</span> (${log.deltaW2 >= 0 ? '+' : ''}${log.deltaW2.toFixed(2)})</div>
          <div>b : ${log.oldB.toFixed(2)} → <span class="text-emerald-300 font-bold">${log.newB.toFixed(2)}</span> (${log.deltaB >= 0 ? '+' : ''}${log.deltaB.toFixed(2)})</div>
        </div>

        <div class="text-[11px] text-indigo-700 bg-indigo-50 p-2 rounded-lg border border-indigo-100 font-medium">
          💡 <strong>해석:</strong> 잘못 분류된 점 방향으로 결정 경계선이 ${log.t === 1 ? '파란 점 쪽으로 당겨졌습니다' : '빨간 점 반대쪽으로 밀려났습니다'}.
        </div>
      </div>
    `;
  }


  // ==========================================================================
  // 8. Chart.js Setup & Updates
  // ==========================================================================
  function initChart() {
    const ctxChart = document.getElementById('lossChart').getContext('2d');

    lossChart = new Chart(ctxChart, {
      type: 'line',
      data: {
        labels: [0],
        datasets: [{
          label: '오분류 개수',
          data: [0],
          borderColor: '#e11d48',
          backgroundColor: 'rgba(225, 29, 72, 0.1)',
          fill: true,
          tension: 0.2,
          pointRadius: 4,
          pointBackgroundColor: '#e11d48'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 200 },
        scales: {
          x: {
            title: { display: true, text: '학습 단계 (Step)', font: { size: 10 } },
            grid: { display: false }
          },
          y: {
            title: { display: true, text: '수치', font: { size: 10 } },
            beginAtZero: true,
            ticks: { precision: 0 }
          }
        },
        plugins: {
          legend: { display: true, position: 'top', labels: { boxWidth: 12, font: { size: 11 } } }
        }
      }
    });
  }

  function updateChart() {
    if (!lossChart) return;

    const labels = history.map(h => h.step);

    if (chartViewMode === 'loss') {
      const lossData = history.map(h => h.errors);
      lossChart.data.labels = labels;
      lossChart.data.datasets = [{
        label: '오분류 개수',
        data: lossData,
        borderColor: '#e11d48',
        backgroundColor: 'rgba(225, 29, 72, 0.1)',
        fill: true,
        tension: 0.2,
        pointRadius: 4,
        pointBackgroundColor: '#e11d48'
      }];
    } else {
      // Weights View
      lossChart.data.labels = labels;
      lossChart.data.datasets = [
        {
          label: 'w1 (x)',
          data: history.map(h => h.w1),
          borderColor: '#4f46e5',
          backgroundColor: 'transparent',
          tension: 0.2,
          pointRadius: 3
        },
        {
          label: 'w2 (y)',
          data: history.map(h => h.w2),
          borderColor: '#2563eb',
          backgroundColor: 'transparent',
          tension: 0.2,
          pointRadius: 3
        },
        {
          label: 'b (Bias)',
          data: history.map(h => h.b),
          borderColor: '#f59e0b',
          backgroundColor: 'transparent',
          tension: 0.2,
          pointRadius: 3
        }
      ];
    }

    lossChart.update();
  }


  // ==========================================================================
  // 9. Interactive Canvas Pointer Handlers (Mouse & Touch)
  // ==========================================================================
  function getPointerPos(e) {
    const rect = canvas.getBoundingClientRect();
    let clientX = e.clientX;
    let clientY = e.clientY;

    if (e.touches && e.touches.length > 0) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    }

    const cx = clientX - rect.left;
    const cy = clientY - rect.top;
    return { cx, cy };
  }

  function findPointAt(cx, cy) {
    const threshold = 16; // Pixel click tolerance
    for (let i = points.length - 1; i >= 0; i--) {
      const p = points[i];
      const pCanvas = mathToCanvas(p.x, p.y);
      const dist = Math.hypot(pCanvas.cx - cx, pCanvas.cy - cy);
      if (dist <= threshold) {
        return p;
      }
    }
    return null;
  }

  function handlePointerDown(e) {
    const { cx, cy } = getPointerPos(e);
    const clickedPoint = findPointAt(cx, cy);

    if (mode === 'select') {
      if (clickedPoint) {
        selectedPointId = clickedPoint.id;
        isDragging = true;
      } else {
        selectedPointId = null;
      }
    } else if (mode === 'blue' || mode === 'red') {
      if (clickedPoint) {
        // Toggle or select existing
        selectedPointId = clickedPoint.id;
      } else {
        // Add new point
        const mathPos = canvasToMath(cx, cy);
        const newPoint = {
          id: 'p_' + Date.now() + '_' + Math.floor(Math.random() * 1000),
          x: mathPos.x,
          y: mathPos.y,
          label: mode === 'blue' ? 1 : -1
        };
        points.push(newPoint);
        selectedPointId = newPoint.id;
      }
    }

    renderCanvas();
    updateStats();
  }

  function handlePointerMove(e) {
    const { cx, cy } = getPointerPos(e);
    const mathPos = canvasToMath(cx, cy);

    // Update Hover Coordinate Badge
    coordDisplay.textContent = `좌표: (${mathPos.x.toFixed(1)}, ${mathPos.y.toFixed(1)})`;

    if (isDragging && selectedPointId) {
      const point = points.find(p => p.id === selectedPointId);
      if (point) {
        point.x = mathPos.x;
        point.y = mathPos.y;
        renderCanvas();
        updateStats();
      }
    }
  }

  function handlePointerUp() {
    isDragging = false;
  }

  function handleDoubleClick(e) {
    const { cx, cy } = getPointerPos(e);
    const clickedPoint = findPointAt(cx, cy);

    if (clickedPoint) {
      points = points.filter(p => p.id !== clickedPoint.id);
      selectedPointId = null;
      renderCanvas();
      updateStats();
    }
  }


  // ==========================================================================
  // 10. Event Listeners Initialization
  // ==========================================================================
  function initEvents() {
    window.addEventListener('resize', resizeCanvas);

    // Canvas Mouse & Touch
    canvas.addEventListener('mousedown', handlePointerDown);
    canvas.addEventListener('mousemove', handlePointerMove);
    window.addEventListener('mouseup', handlePointerUp);
    canvas.addEventListener('dblclick', handleDoubleClick);

    canvas.addEventListener('touchstart', (e) => {
      e.preventDefault();
      handlePointerDown(e);
    }, { passive: false });

    canvas.addEventListener('touchmove', (e) => {
      e.preventDefault();
      handlePointerMove(e);
    }, { passive: false });

    canvas.addEventListener('touchend', (e) => {
      e.preventDefault();
      handlePointerUp();
    });

    // Mode Buttons
    modeBlueBtn.addEventListener('click', () => {
      mode = 'blue';
      modeBlueBtn.className = 'px-3 py-1.5 text-xs font-bold rounded-lg transition-all flex items-center gap-1.5 bg-blue-600 text-white shadow-sm';
      modeRedBtn.className = 'px-3 py-1.5 text-xs font-bold rounded-lg transition-all flex items-center gap-1.5 text-slate-600 hover:text-slate-900';
      modeSelectBtn.className = 'px-3 py-1.5 text-xs font-bold rounded-lg transition-all flex items-center gap-1.5 text-slate-600 hover:text-slate-900';
    });

    modeRedBtn.addEventListener('click', () => {
      mode = 'red';
      modeRedBtn.className = 'px-3 py-1.5 text-xs font-bold rounded-lg transition-all flex items-center gap-1.5 bg-rose-600 text-white shadow-sm';
      modeBlueBtn.className = 'px-3 py-1.5 text-xs font-bold rounded-lg transition-all flex items-center gap-1.5 text-slate-600 hover:text-slate-900';
      modeSelectBtn.className = 'px-3 py-1.5 text-xs font-bold rounded-lg transition-all flex items-center gap-1.5 text-slate-600 hover:text-slate-900';
    });

    modeSelectBtn.addEventListener('click', () => {
      mode = 'select';
      modeSelectBtn.className = 'px-3 py-1.5 text-xs font-bold rounded-lg transition-all flex items-center gap-1.5 bg-slate-800 text-white shadow-sm';
      modeBlueBtn.className = 'px-3 py-1.5 text-xs font-bold rounded-lg transition-all flex items-center gap-1.5 text-slate-600 hover:text-slate-900';
      modeRedBtn.className = 'px-3 py-1.5 text-xs font-bold rounded-lg transition-all flex items-center gap-1.5 text-slate-600 hover:text-slate-900';
    });

    // Presets Dropdown
    presetSelect.addEventListener('change', (e) => {
      loadPreset(e.target.value);
    });

    // Learning Rate Slider
    learningRateInput.addEventListener('input', (e) => {
      learningRate = parseFloat(e.target.value);
      lrValDisplay.textContent = learningRate.toFixed(2);
    });

    // Quick LR Buttons
    document.querySelectorAll('.lrQuickBtn').forEach(btn => {
      btn.addEventListener('click', () => {
        const val = parseFloat(btn.dataset.val);
        learningRate = val;
        learningRateInput.value = val;
        lrValDisplay.textContent = val.toFixed(2);
      });
    });

    // Speed Slider
    speedInput.addEventListener('input', (e) => {
      speedMs = parseInt(e.target.value, 10);
      speedValDisplay.textContent = `${speedMs}ms`;
      if (isPlaying) {
        stopAutoPlay();
        startAutoPlay();
      }
    });

    // Execution Buttons
    autoPlayBtn.addEventListener('click', () => {
      if (isPlaying) {
        stopAutoPlay();
      } else {
        startAutoPlay();
      }
    });

    stepBtn.addEventListener('click', () => {
      stopAutoPlay();
      trainSingleStep();
    });

    resetWeightsBtn.addEventListener('click', resetWeights);

    clearPointsBtn.addEventListener('click', () => {
      stopAutoPlay();
      points = [];
      presetSelect.value = 'custom';
      resetWeights();
    });

    // Chart View Toggle
    chartTabLoss.addEventListener('click', () => {
      chartViewMode = 'loss';
      chartTabLoss.className = 'px-2.5 py-1 rounded-md transition bg-white text-indigo-600 shadow-sm';
      chartTabWeights.className = 'px-2.5 py-1 rounded-md transition text-slate-600 hover:text-slate-900';
      updateChart();
    });

    chartTabWeights.addEventListener('click', () => {
      chartViewMode = 'weights';
      chartTabWeights.className = 'px-2.5 py-1 rounded-md transition bg-white text-indigo-600 shadow-sm';
      chartTabLoss.className = 'px-2.5 py-1 rounded-md transition text-slate-600 hover:text-slate-900';
      updateChart();
    });

    // Guide Modal Handlers
    guideBtn.addEventListener('click', () => {
      guideModal.classList.remove('hidden');
    });

    closeGuideBtn.addEventListener('click', () => {
      guideModal.classList.add('hidden');
    });

    confirmGuideBtn.addEventListener('click', () => {
      guideModal.classList.add('hidden');
    });

    guideModal.addEventListener('click', (e) => {
      if (e.target === guideModal) {
        guideModal.classList.add('hidden');
      }
    });
  }


  // ==========================================================================
  // 11. App Initialization
  // ==========================================================================
  function init() {
    initChart();
    initEvents();
    resizeCanvas();

    // Default Preset: Diagonal Linearly Separable Dataset
    presetSelect.value = 'diagonal';
    loadPreset('diagonal');
  }

  // Run on DOM Ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
