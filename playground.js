/**
 * Surya Prakash - AI/ML Portfolio
 * AI Sandbox: Interactive Neural Network Visualizer
 * Pure JS Multi-Layer Perceptron (MLP) with Backpropagation and HTML5 Canvas Graphics
 */

class NeuralNetwork {
  constructor(layerSizes, activationName = 'tanh') {
    this.layerSizes = layerSizes; // e.g., [2, 4, 3, 1]
    this.activationName = activationName;
    this.weights = []; // Array of matrices for weights between layers
    this.biases = [];  // Array of vectors for biases of layers

    // Initialize weights and biases
    for (let i = 0; i < layerSizes.length - 1; i++) {
      const rows = layerSizes[i + 1];
      const cols = layerSizes[i];
      
      // Xavier / Glorot-like initialization
      const limit = Math.sqrt(6 / (rows + cols));
      
      const layerWeights = [];
      for (let r = 0; r < rows; r++) {
        const row = [];
        for (let c = 0; c < cols; c++) {
          row.push((Math.random() * 2 - 1) * limit);
        }
        layerWeights.push(row);
      }
      this.weights.push(layerWeights);

      const layerBiases = [];
      for (let r = 0; r < rows; r++) {
        layerBiases.push(0.0); // Start biases at 0
      }
      this.biases.push(layerBiases);
    }
  }

  // Get activation functions
  getActivation() {
    const act = this.activationName.toLowerCase();
    if (act === 'relu') {
      return {
        f: x => Math.max(0, x),
        df: y => y > 0 ? 1 : 0
      };
    } else if (act === 'sigmoid') {
      return {
        f: x => 1 / (1 + Math.exp(-x)),
        df: y => y * (1 - y)
      };
    } else { // default 'tanh'
      return {
        f: x => Math.tanh(x),
        df: y => 1 - y * y
      };
    }
  }

  // Final layer always uses Sigmoid for binary probability output
  getOutputActivation() {
    return {
      f: x => 1 / (1 + Math.exp(-x)),
      df: y => y * (1 - y)
    };
  }

  // Forward propagate input, returns activations and raw inputs (z) for all layers
  forward(input) {
    const activations = [input];
    const zs = [];
    const hiddenAct = this.getActivation();
    const outputAct = this.getOutputActivation();

    let current = input;
    for (let i = 0; i < this.weights.length; i++) {
      const W = this.weights[i];
      const b = this.biases[i];
      const next = [];
      const z = [];

      const actFunc = (i === this.weights.length - 1) ? outputAct.f : hiddenAct.f;

      for (let r = 0; r < W.length; r++) {
        let sum = b[r];
        for (let c = 0; c < W[r].length; c++) {
          sum += W[r][c] * current[c];
        }
        z.push(sum);
        next.push(actFunc(sum));
      }
      zs.push(z);
      current = next;
      activations.push(current);
    }

    return { activations, zs };
  }

  // Backpropagation: computes gradients and performs weights update
  trainStep(input, target, learningRate) {
    const { activations, zs } = this.forward(input);
    const hiddenAct = this.getActivation();
    const outputAct = this.getOutputActivation();
    
    // Store gradients
    const weightGradients = [];
    const biasGradients = [];
    
    // Initialize gradient structures
    for (let i = 0; i < this.weights.length; i++) {
      weightGradients.push(this.weights[i].map(row => row.map(() => 0)));
      biasGradients.push(this.biases[i].map(() => 0));
    }

    const L = this.weights.length;
    let deltas = [];

    // Output layer error delta (Mean Squared Error derivative)
    // dE/dz = (a - t) * f'(z)
    const outputActivation = activations[L];
    const outputDelta = [];
    for (let i = 0; i < outputActivation.length; i++) {
      const error = outputActivation[i] - target[i];
      outputDelta.push(error * outputAct.df(outputActivation[i]));
    }
    deltas[L - 1] = outputDelta;

    // Backpropagate error deltas
    for (let l = L - 2; l >= 0; l--) {
      const layerDelta = [];
      const nextDelta = deltas[l + 1];
      const W_next = this.weights[l + 1];
      const layerActivation = activations[l + 1];

      for (let i = 0; i < W_next[0].length; i++) {
        let sum = 0;
        for (let j = 0; j < W_next.length; j++) {
          sum += W_next[j][i] * nextDelta[j];
        }
        layerDelta.push(sum * hiddenAct.df(layerActivation[i]));
      }
      deltas[l] = layerDelta;
    }

    // Apply Gradient Descent updates
    for (let l = 0; l < L; l++) {
      const W = this.weights[l];
      const b = this.biases[l];
      const delta = deltas[l];
      const prevActivation = activations[l];

      for (let r = 0; r < W.length; r++) {
        b[r] -= learningRate * delta[r];
        // Gradient clip to prevent exploding weights
        b[r] = Math.max(-5, Math.min(5, b[r]));

        for (let c = 0; c < W[r].length; c++) {
          W[r][c] -= learningRate * delta[r] * prevActivation[c];
          W[r][c] = Math.max(-5, Math.min(5, W[r][c]));
        }
      }
    }

    // Calculate prediction loss (MSE)
    let loss = 0;
    for (let i = 0; i < target.length; i++) {
      loss += 0.5 * Math.pow(outputActivation[i] - target[i], 2);
    }
    return loss;
  }
}

/* ==========================================================================
   Playground Orchestrator & UI Management
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  // Config state
  let datasetType = 'circle'; // 'circle' or 'xor'
  let hiddenLayers = [4, 2];   // Default layer neuron counts
  let activationName = 'tanh';
  let learningRate = 0.05;
  let network = null;
  let isTraining = false;
  let epoch = 0;
  let dataset = [];
  let lossHistory = [];
  let trainRequestId = null;

  // Visualizer drawing variables
  let particles = [];
  
  // DOM Elements
  const networkCanvas = document.getElementById('network-canvas');
  const boundaryCanvas = document.getElementById('boundary-canvas');
  const btnTrain = document.getElementById('btn-train-sandbox');
  const btnReset = document.getElementById('btn-reset-sandbox');
  const datasetSelect = document.getElementById('select-dataset');
  const activationSelect = document.getElementById('select-activation');
  const lrSlider = document.getElementById('slider-lr');
  const lrValLabel = document.getElementById('val-lr');
  const epochLabel = document.getElementById('label-epoch');
  const lossLabel = document.getElementById('label-loss');
  const layersList = document.getElementById('layers-list');
  const btnAddLayer = document.getElementById('btn-add-layer');

  if (!networkCanvas || !boundaryCanvas) return;

  const netCtx = networkCanvas.getContext('2d');
  const boundCtx = boundaryCanvas.getContext('2d');

  // Handle high DPI displays
  function setupCanvasDPI(canvas, context) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    // Set explicit transformation to prevent compounding scales on resize
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function initCanvases() {
    setupCanvasDPI(networkCanvas, netCtx);
    setupCanvasDPI(boundaryCanvas, boundCtx);
  }

  // Populate synthetic classification dataset
  function generateDataset() {
    dataset = [];
    const count = 100;
    
    // Seeded random helper for deterministic layout
    let seed = 12345;
    function seededRandom() {
      const x = Math.sin(seed++) * 10000;
      return x - Math.floor(x);
    }

    if (datasetType === 'circle') {
      for (let i = 0; i < count; i++) {
        const r = seededRandom() * 0.9;
        const theta = seededRandom() * Math.PI * 2;
        const x = r * Math.cos(theta);
        const y = r * Math.sin(theta);
        // Circle border at radius 0.5
        const label = (r < 0.5) ? 1.0 : 0.0;
        dataset.push({ input: [x, y], target: [label] });
      }
    } else if (datasetType === 'xor') {
      for (let i = 0; i < count; i++) {
        // Add small jitter so points aren't exactly on 0
        const x = (seededRandom() * 2 - 1) * 0.9;
        const y = (seededRandom() * 2 - 1) * 0.9;
        // XOR quadrant logic: quadrant 1 & 3 are 1, quadrant 2 & 4 are 0
        // adding a little margin so it's clean
        const label = (x * y > 0) ? 1.0 : 0.0;
        dataset.push({ input: [x, y], target: [label] });
      }
    }
  }

  // Instantiates the model matching UI layers configuration
  function resetNetwork() {
    isTraining = false;
    btnTrain.innerHTML = '<i data-lucide="play" class="w-4 h-4"></i><span>Train Sandbox</span>';
    lucide.createIcons();
    
    epoch = 0;
    lossHistory = [];
    epochLabel.textContent = '0';
    lossLabel.textContent = '0.000';
    
    // Architecture: Inputs (2) -> Hidden Layers -> Outputs (1)
    const fullLayers = [2, ...hiddenLayers, 1];
    network = new NeuralNetwork(fullLayers, activationName);
    
    generateDataset();
    initializeSynapseParticles();
    drawAll();
  }

  // Setup flowing synapse signals
  function initializeSynapseParticles() {
    particles = [];
    const fullLayers = [2, ...hiddenLayers, 1];
    
    // Precompute node locations for synapse flows
    const layout = getNodeLayout(fullLayers);
    
    for (let l = 0; l < layout.length - 1; l++) {
      const currentLayer = layout[l];
      const nextLayer = layout[l + 1];
      
      for (let i = 0; i < currentLayer.length; i++) {
        for (let j = 0; j < nextLayer.length; j++) {
          // Generate 2 flowing dots per connection line
          particles.push({
            layer: l,
            fromIdx: i,
            toIdx: j,
            progress: Math.random(), // randomize phase
            speed: 0.008 + Math.random() * 0.006
          });
        }
      }
    }
  }

  // Get pixel coordinate locations for all nodes on networkCanvas
  function getNodeLayout(layers) {
    const rect = networkCanvas.getBoundingClientRect();
    const width = rect.width || 300;
    const height = rect.height || 300;
    
    const layerSpacing = width / (layers.length - 1 || 1);
    const layout = [];

    for (let l = 0; l < layers.length; l++) {
      const nodeCount = layers[l];
      const x = l * layerSpacing;
      
      const layerNodes = [];
      const nodeSpacing = height / (nodeCount + 1 || 1);
      
      for (let n = 0; n < nodeCount; n++) {
        const y = (n + 1) * nodeSpacing;
        layerNodes.push({ x, y });
      }
      layout.push(layerNodes);
    }
    return layout;
  }

  // Renders the decision boundary grids and classification scatter dots
  function drawBoundary() {
    const rect = boundaryCanvas.getBoundingClientRect();
    const w = rect.width || 200;
    const h = rect.height || 200;
    
    boundCtx.clearRect(0, 0, w, h);

    // 1. Render background classification boundary grid
    const gridSize = 45; // grid resolution
    const cellW = w / gridSize;
    const cellH = h / gridSize;

    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        // Map pixel coordinates to normalized model space [-1, 1]
        const nx = (i / gridSize) * 2 - 1;
        const ny = (j / gridSize) * 2 - 1;
        
        const { activations } = network.forward([nx, ny]);
        const pred = activations[activations.length - 1][0];

        // Draw color interpolation based on prediction output probability
        // Blue (probability -> 1), Pink (probability -> 0)
        // Set low opacity background overlay
        let fillStyle = '';
        if (pred > 0.5) {
          const intensity = (pred - 0.5) * 2; // scale [0, 1]
          fillStyle = `rgba(99, 102, 241, ${0.12 + intensity * 0.28})`; // Slate Indigo
        } else {
          const intensity = (0.5 - pred) * 2; // scale [0, 1]
          fillStyle = `rgba(236, 72, 153, ${0.12 + intensity * 0.28})`; // Hot Pink
        }
        
        boundCtx.fillStyle = fillStyle;
        boundCtx.fillRect(i * cellW, j * cellH, cellW + 0.5, cellH + 0.5);
      }
    }

    // 2. Draw Dataset points
    dataset.forEach(pt => {
      // Map [-1, 1] space back to Canvas layout pixel spaces
      const px = ((pt.input[0] + 1) / 2) * w;
      const py = ((pt.input[1] + 1) / 2) * h;
      
      boundCtx.beginPath();
      boundCtx.arc(px, py, 5, 0, Math.PI * 2);
      
      if (pt.target[0] === 1.0) {
        boundCtx.fillStyle = '#6366f1'; // Glowing Blue node
        boundCtx.shadowColor = '#6366f1';
        boundCtx.strokeStyle = '#ffffff';
      } else {
        boundCtx.fillStyle = '#ec4899'; // Glowing Pink node
        boundCtx.shadowColor = '#ec4899';
        boundCtx.strokeStyle = '#ffffff';
      }
      
      boundCtx.lineWidth = 1.2;
      boundCtx.shadowBlur = 4;
      boundCtx.fill();
      boundCtx.stroke();
      
      // Reset shadow blur
      boundCtx.shadowBlur = 0;
    });
  }

  // Draw Neurons and Synapse pathways (with weights & signals)
  function drawNetwork() {
    const rect = networkCanvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    
    netCtx.clearRect(0, 0, w, h);
    
    const fullLayers = [2, ...hiddenLayers, 1];
    const layout = getNodeLayout(fullLayers);

    // 1. Draw Synapses (Lines colored & sized by weights value)
    for (let l = 0; l < layout.length - 1; l++) {
      const currentLayer = layout[l];
      const nextLayer = layout[l + 1];
      const weights = network.weights[l];

      for (let i = 0; i < currentLayer.length; i++) {
        for (let j = 0; j < nextLayer.length; j++) {
          const from = currentLayer[i];
          const to = nextLayer[j];
          const weight = weights[j][i];
          
          netCtx.beginPath();
          netCtx.moveTo(from.x, from.y);
          netCtx.lineTo(to.x, to.y);
          
          // Color weights: indigo positive, pink negative
          const maxWeight = 3.0; // limit scaling
          const absWeight = Math.min(Math.abs(weight), maxWeight);
          const opacity = 0.08 + (absWeight / maxWeight) * 0.65;
          
          netCtx.strokeStyle = weight >= 0 
            ? `rgba(99, 102, 241, ${opacity})`  // Indigo
            : `rgba(236, 72, 153, ${opacity})`; // Pink
          
          netCtx.lineWidth = 0.5 + (absWeight / maxWeight) * 3;
          netCtx.stroke();
        }
      }
    }

    // 2. Draw Moving Flow Particles
    particles.forEach(pt => {
      const fromNode = layout[pt.layer][pt.fromIdx];
      const toNode = layout[pt.layer + 1][pt.toIdx];
      
      // Interpolate position
      const px = fromNode.x + (toNode.x - fromNode.x) * pt.progress;
      const py = fromNode.y + (toNode.y - fromNode.y) * pt.progress;
      
      // Determine weight color to match flow particle
      const weight = network.weights[pt.layer][pt.toIdx][pt.fromIdx];
      
      netCtx.beginPath();
      netCtx.arc(px, py, 2.5, 0, Math.PI * 2);
      netCtx.fillStyle = weight >= 0 ? '#06b6d4' : '#a855f7'; // Cyan for positive signal, Purple for negative
      netCtx.shadowColor = weight >= 0 ? '#06b6d4' : '#a855f7';
      netCtx.shadowBlur = 6;
      netCtx.fill();
      netCtx.shadowBlur = 0;

      // Update flow particle location
      if (isTraining) {
        pt.progress += pt.speed;
        if (pt.progress > 1.0) {
          pt.progress = 0;
        }
      }
    });

    // 3. Draw Neuron Nodes
    for (let l = 0; l < layout.length; l++) {
      const nodes = layout[l];
      
      nodes.forEach((node, idx) => {
        netCtx.beginPath();
        netCtx.arc(node.x, node.y, 11, 0, Math.PI * 2);
        
        // Coloring classes
        let fillStyle = '#0f172a'; // dark inside
        let borderStyle = '#334155';
        
        if (l === 0) { // Input
          borderStyle = '#06b6d4'; // Cyan
        } else if (l === layout.length - 1) { // Output
          borderStyle = '#a855f7'; // Purple
        } else { // Hidden layers
          borderStyle = '#6366f1'; // Indigo
        }
        
        netCtx.fillStyle = fillStyle;
        netCtx.strokeStyle = borderStyle;
        netCtx.lineWidth = 2.5;
        netCtx.shadowColor = borderStyle;
        netCtx.shadowBlur = 4;
        netCtx.fill();
        netCtx.stroke();
        netCtx.shadowBlur = 0;
        
        // Draw layer info labels inside the neurons (subtle text indicators)
        netCtx.fillStyle = '#94a3b8';
        netCtx.font = 'bold 8px monospace';
        netCtx.textAlign = 'center';
        netCtx.textBaseline = 'middle';
        
        let label = '';
        if (l === 0) label = `X${idx + 1}`;
        else if (l === layout.length - 1) label = 'Y';
        else label = `H${l}_${idx + 1}`;
        
        netCtx.fillText(label, node.x, node.y);
      });
    }
  }

  // Draw both network and decision boundaries
  function drawAll() {
    drawBoundary();
    drawNetwork();
  }

  // Training epoch calculations loop
  function trainLoop() {
    if (!isTraining) return;

    let batchLossSum = 0;
    const batchRuns = 8; // Number of training steps per animation frame

    for (let step = 0; step < batchRuns; step++) {
      // Shuffle training batch
      const shuffled = [...dataset].sort(() => Math.random() - 0.5);
      let epochLoss = 0;

      shuffled.forEach(pt => {
        const lossVal = network.trainStep(pt.input, pt.target, learningRate);
        epochLoss += lossVal;
      });

      epoch++;
      batchLossSum += (epochLoss / dataset.length);
    }

    const averageLoss = batchLossSum / batchRuns;
    
    // Update Stats Label
    epochLabel.textContent = epoch.toString();
    lossLabel.textContent = averageLoss.toFixed(4);

    drawAll();
    trainRequestId = requestAnimationFrame(trainLoop);
  }

  // Trigger training pause/run state
  function toggleTraining() {
    if (isTraining) {
      isTraining = false;
      cancelAnimationFrame(trainRequestId);
      btnTrain.innerHTML = '<i data-lucide="play" class="w-4 h-4"></i><span>Train Sandbox</span>';
    } else {
      isTraining = true;
      btnTrain.innerHTML = '<i data-lucide="pause" class="w-4 h-4"></i><span>Pause Training</span>';
      trainLoop();
    }
    lucide.createIcons();
  }

  // UI Component rendering for hidden layer configuration controllers
  function renderHiddenLayersUI() {
    layersList.innerHTML = '';
    
    hiddenLayers.forEach((neurons, idx) => {
      const item = document.createElement('div');
      item.className = 'flex items-center justify-between p-3 rounded-lg bg-slate-900/80 border border-slate-800 text-xs';
      item.innerHTML = `
        <span class="font-medium text-slate-300">Layer ${idx + 1} (${neurons} nodes)</span>
        <div class="flex items-center gap-1.5">
          <button class="btn-node-sub w-6 h-6 rounded bg-slate-800 hover:bg-slate-750 text-slate-300 border border-slate-750 flex items-center justify-center font-bold text-xs" data-idx="${idx}">-</button>
          <button class="btn-node-add w-6 h-6 rounded bg-slate-800 hover:bg-slate-750 text-slate-300 border border-slate-750 flex items-center justify-center font-bold text-xs" data-idx="${idx}">+</button>
          <button class="btn-layer-del ml-2 w-6 h-6 rounded bg-accent-pink/10 hover:bg-accent-pink/20 text-accent-pink border border-accent-pink/20 flex items-center justify-center" data-idx="${idx}">
            <i data-lucide="trash-2" class="w-3.5 h-3.5"></i>
          </button>
        </div>
      `;
      layersList.appendChild(item);
    });

    lucide.createIcons();
    
    // Bind dynamic adjustments inside layers count
    layersList.querySelectorAll('.btn-node-add').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const idx = parseInt(e.target.getAttribute('data-idx'), 10);
        if (hiddenLayers[idx] < 6) {
          hiddenLayers[idx]++;
          renderHiddenLayersUI();
          resetNetwork();
        }
      });
    });

    layersList.querySelectorAll('.btn-node-sub').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const idx = parseInt(e.target.getAttribute('data-idx'), 10);
        if (hiddenLayers[idx] > 1) {
          hiddenLayers[idx]--;
          renderHiddenLayersUI();
          resetNetwork();
        }
      });
    });

    layersList.querySelectorAll('.btn-layer-del').forEach(btn => {
      const btnActual = btn.tagName === 'BUTTON' ? btn : btn.closest('button');
      btnActual.addEventListener('click', (e) => {
        const idx = parseInt(btnActual.getAttribute('data-idx'), 10);
        hiddenLayers.splice(idx, 1);
        renderHiddenLayersUI();
        resetNetwork();
      });
    });

    // Disable adding layers if limit reached
    if (hiddenLayers.length >= 3) {
      btnAddLayer.disabled = true;
      btnAddLayer.classList.add('opacity-40', 'cursor-not-allowed');
    } else {
      btnAddLayer.disabled = false;
      btnAddLayer.classList.remove('opacity-40', 'cursor-not-allowed');
    }
  }

  // EVENT LISTENERS
  btnTrain.addEventListener('click', toggleTraining);
  btnReset.addEventListener('click', resetNetwork);

  datasetSelect.addEventListener('change', (e) => {
    datasetType = e.target.value;
    resetNetwork();
  });

  activationSelect.addEventListener('change', (e) => {
    activationName = e.target.value;
    resetNetwork();
  });

  lrSlider.addEventListener('input', (e) => {
    learningRate = parseFloat(e.target.value);
    lrValLabel.textContent = learningRate.toString();
  });

  btnAddLayer.addEventListener('click', () => {
    if (hiddenLayers.length < 3) {
      hiddenLayers.push(3); // Start new layer with 3 neurons
      renderHiddenLayersUI();
      resetNetwork();
    }
  });

  // Handle screen resize responsive DPI calculations
  window.addEventListener('resize', () => {
    initCanvases();
    drawAll();
  });

  // Initial startup execution
  renderHiddenLayersUI();
  initCanvases();
  resetNetwork();
});
