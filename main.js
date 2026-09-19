/* ============================================================
   Animal Sound Identifier — high-accuracy edition
   ------------------------------------------------------------
   Accuracy techniques used (code-side):
   1. CONFIDENCE THRESHOLD  — ignore weak predictions.
   2. TEMPORAL SMOOTHING    — average scores over a rolling
                              window of frames (majority voting),
                              so one noisy frame can't flip result.
   3. CONSECUTIVE-FRAME LOCK— a label must win several frames in
                              a row before it's accepted.
   4. OVERLAP FACTOR        — classify more often per second for
                              more evidence.
   5. EVENT DEBOUNCING      — one sustained bark = ONE detection,
                              not 20.
   For the last mile to 99%: retrain the Teachable Machine model
   with 100+ varied samples per class AND a "Background Noise"
   class, then paste the new model URL below.
   ============================================================ */

// ------------------------- CONFIG ---------------------------
const MODEL_URL = 'https://teachablemachine.withgoogle.com/models/u02fOYRBx/model.json';

const CONFIDENCE_THRESHOLD = 0.85; // ignore predictions below 85%
const SMOOTHING_WINDOW     = 8;    // frames averaged for a decision
const CONSECUTIVE_NEEDED   = 3;    // frames in a row a label must win
const EVENT_COOLDOWN_MS    = 2500; // min gap between two counted events
const OVERLAP_FACTOR       = 0.75; // 0..1 — higher = more frequent predictions

// Map model labels -> UI data. Adjust label keys if your model
// uses different class names (check the console output).
const ANIMALS = {
  'Barking': { name: 'Dog 🐶', img: 'Bark.gif', counterId: 'dog_count' },
  'Meowing': { name: 'Cat 🐱', img: 'Meow.gif', counterId: 'cat_count' }
};

// ------------------------- STATE ----------------------------
let classifier = null;
let started = false;

let scoreHistory = [];        // rolling window of {label: confidence} frames
let consecutiveLabel = null;  // label currently on a winning streak
let consecutiveCount = 0;
let currentStable = null;     // the label we currently display
let lastEventTime = 0;        // for debouncing counted detections
const counts = { dog_count: 0, cat_count: 0 };

// ------------------------- START ----------------------------
function startClassification() {
  if (started) return;

  setStatus('Requesting microphone…');

  navigator.mediaDevices.getUserMedia({ audio: true })
    .then(() => {
      setStatus('Loading model…');
      const options = {
        probabilityThreshold: 0,        // we do our own thresholding
        overlapFactor: OVERLAP_FACTOR   // predict more often
      };
      classifier = ml5.soundClassifier(MODEL_URL, options, modelReady);
    })
    .catch(err => {
      console.error(err);
      setStatus('❌ Microphone access denied. Please allow the mic and reload.');
    });
}

function modelReady() {
  started = true;
  setStatus('✅ Listening… play a dog or cat sound!');
  document.getElementById('start_btn').classList.add('listening');
  document.getElementById('start_btn').innerHTML =
    '<span class="btn-icon">🔊</span> Listening…';
  classifier.classify(gotResults);
}

// ---------------------- CLASSIFICATION ----------------------
function gotResults(error, results) {
  if (error) { console.error(error); return; }

  // ---- 1. Render raw live prediction bars (transparency for the user)
  renderLiveBars(results);

  // ---- 2. Push this frame into the rolling window
  const frame = {};
  results.forEach(r => { frame[r.label] = r.confidence; });
  scoreHistory.push(frame);
  if (scoreHistory.length > SMOOTHING_WINDOW) scoreHistory.shift();

  // ---- 3. Average confidences across the window (temporal smoothing)
  const avg = {};
  results.forEach(r => {
    let sum = 0;
    scoreHistory.forEach(f => { sum += (f[r.label] || 0); });
    avg[r.label] = sum / scoreHistory.length;
  });

  // ---- 4. Find the smoothed winner
  let bestLabel = null, bestScore = 0;
  Object.keys(avg).forEach(label => {
    if (avg[label] > bestScore) { bestScore = avg[label]; bestLabel = label; }
  });

  // ---- 5. Apply confidence threshold
  const isAnimal = ANIMALS.hasOwnProperty(bestLabel);
  const passes = bestScore >= CONFIDENCE_THRESHOLD && isAnimal;

  // ---- 6. Consecutive-frame lock: label must keep winning
  if (passes) {
    if (bestLabel === consecutiveLabel) {
      consecutiveCount++;
    } else {
      consecutiveLabel = bestLabel;
      consecutiveCount = 1;
    }
  } else {
    consecutiveLabel = null;
    consecutiveCount = 0;
  }

  if (consecutiveCount >= CONSECUTIVE_NEEDED) {
    acceptDetection(bestLabel, bestScore);
  } else if (!passes && bestScore < CONFIDENCE_THRESHOLD * 0.6) {
    // clearly quiet / uncertain — go back to idle
    resetToIdle();
  }

  updateConfidenceBar(bestScore, passes);
}

// -------------------- ACCEPT A DETECTION --------------------
function acceptDetection(label, score) {
  const animal = ANIMALS[label];
  const pct = Math.round(score * 100);

  document.getElementById('result_label').innerHTML =
    'Detected voice is of — <strong>' + animal.name + '</strong> (' + pct + '%)';
  document.getElementById('animal_image').src = animal.img;

  // Debounce: one sustained sound == one counted event
  const now = Date.now();
  if (label !== currentStable || (now - lastEventTime) > EVENT_COOLDOWN_MS) {
    counts[animal.counterId]++;
    document.getElementById(animal.counterId).textContent = counts[animal.counterId];
    addLogEntry(animal.name, pct);
    lastEventTime = now;
  }
  currentStable = label;
}

function resetToIdle() {
  if (currentStable !== null) {
    currentStable = null;
    document.getElementById('result_label').innerHTML = 'I can hear — nothing yet';
    document.getElementById('animal_image').src = 'listen.gif';
  }
}

// ------------------------- UI HELPERS ------------------------
function setStatus(msg) {
  document.getElementById('status_text').textContent = msg;
}

function renderLiveBars(results) {
  const wrap = document.getElementById('live_bars');
  wrap.innerHTML = '';
  results
    .slice()
    .sort((a, b) => b.confidence - a.confidence)
    .forEach(r => {
      const pct = Math.round(r.confidence * 100);
      const row = document.createElement('div');
      row.className = 'bar-row';
      row.innerHTML =
        '<span class="bar-label">' + r.label + '</span>' +
        '<div class="bar-track"><div class="bar-fill" style="width:' + pct + '%"></div></div>' +
        '<span class="bar-pct">' + pct + '%</span>';
      wrap.appendChild(row);
    });
}

function updateConfidenceBar(score, passes) {
  const pct = Math.round(score * 100);
  const bar = document.getElementById('confidence_bar');
  bar.style.width = pct + '%';
  bar.className = 'confidence-fill ' + (passes ? 'good' : 'low');
  document.getElementById('confidence_text').textContent = pct + '%';
}

function addLogEntry(name, pct) {
  const log = document.getElementById('detection_log');
  const placeholder = log.querySelector('.muted');
  if (placeholder) placeholder.remove();

  const li = document.createElement('li');
  const t = new Date().toLocaleTimeString();
  li.textContent = t + ' — ' + name + ' (' + pct + '%)';
  log.prepend(li);

  while (log.children.length > 8) log.removeChild(log.lastChild);
}
