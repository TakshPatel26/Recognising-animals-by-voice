/* ==================================================================
   Animal Sound Identifier — powered by Google YAMNet (TensorFlow.js)
   ------------------------------------------------------------------
   YAMNet is a deep net trained by Google on the AudioSet corpus
   (millions of labeled YouTube audio clips, 521 sound classes).
   Because it has seen thousands of real barks, meows, roars, moos,
   chirps etc., it is far more accurate than a small self-trained
   Teachable Machine model.

   Pipeline:
     Microphone (getUserMedia)
       -> AudioContext resampled to 16 kHz mono
       -> rolling 0.975 s waveform buffer (15600 samples)
       -> YAMNet graph model -> 521 class scores
       -> map AudioSet classes to our animals
       -> temporal smoothing + confidence threshold + debouncing
       -> UI
   ================================================================== */

// ------------------------------ CONFIG -----------------------------
/* Model sources, tried in order. The tfhub.dev URL is the official one
   (it redirects to Kaggle Models storage since the TF Hub migration). */
const MODEL_SOURCES = [
  { url: 'https://tfhub.dev/google/tfjs-model/yamnet/tfjs/1',
    options: { fromTFHub: true } },
  { url: 'https://www.kaggle.com/models/google/yamnet/TfJs/tfjs/1',
    options: { fromTFHub: true } }
];

const SAMPLE_RATE          = 16000;   // YAMNet expects 16 kHz mono
const INPUT_SAMPLES        = 15600;   // 0.975 s window
const INFER_INTERVAL_MS    = 500;     // run inference twice per second
const CONFIDENCE_THRESHOLD = 0.35;    // YAMNet scores are conservative;
                                      // 0.35 on an animal class is a strong signal
const SMOOTHING_WINDOW     = 4;       // frames averaged (≈2 s of audio)
const CONSECUTIVE_NEEDED   = 2;       // frames in a row a label must win
const EVENT_COOLDOWN_MS    = 3000;    // min gap between counted events

/* Map AudioSet class indices -> our animals.
   Indices from the official yamnet_class_map.csv. */
const ANIMALS = {
  dog:     { name: 'Dog',     emoji: '🐶', img: 'Bark.gif',
             classes: { 69: 'Dog', 70: 'Bark', 71: 'Yip', 72: 'Howl',
                        73: 'Bow-wow', 74: 'Growling', 75: 'Whimper (dog)' } },
  cat:     { name: 'Cat',     emoji: '🐱', img: 'Meow.gif',
             classes: { 76: 'Cat', 77: 'Purr', 78: 'Meow', 79: 'Hiss', 80: 'Caterwaul' } },
  lion:    { name: 'Lion',    emoji: '🦁', img: 'Lion.png',
             classes: { 96: 'Roaring cats (lions, tigers)', 97: 'Growling' } },
  bird:    { name: 'Bird',    emoji: '🐦', img: null,
             classes: { 106: 'Bird', 107: 'Bird vocalization, bird call, bird song',
                        108: 'Chirp, tweet', 109: 'Squawk', 111: 'Coo',
                        112: 'Crow', 113: 'Caw' } },
  rooster: { name: 'Rooster', emoji: '🐓', img: null,
             classes: { 99: 'Chicken, rooster', 100: 'Cluck',
                        101: 'Crowing, cock-a-doodle-doo' } },
  horse:   { name: 'Horse',   emoji: '🐴', img: null,
             classes: { 84: 'Horse', 85: 'Clip-clop', 86: 'Neigh, whinny' } },
  cow:     { name: 'Cow',     emoji: '🐮', img: null,
             classes: { 87: 'Cattle, bovinae', 88: 'Moo', 89: 'Cowbell' } },
  pig:     { name: 'Pig',     emoji: '🐷', img: null,
             classes: { 90: 'Pig', 91: 'Oink' } },
  sheep:   { name: 'Sheep/Goat', emoji: '🐑', img: null,
             classes: { 92: 'Goat', 93: 'Bleat', 94: 'Sheep' } },
  frog:    { name: 'Frog',    emoji: '🐸', img: null,
             classes: { 124: 'Frog', 125: 'Croak' } }
};

// Friendly names for common non-animal classes (shown in live bars)
const EXTRA_CLASS_NAMES = {
  0: 'Speech', 1: 'Child speech', 2: 'Conversation', 3: 'Narration',
  5: 'Shout', 13: 'Laughter', 132: 'Music', 494: 'Silence',
  36: 'Whistling', 47: 'Sneeze', 42: 'Cough', 67: 'Animal sounds',
  68: 'Domestic animals, pets', 81: 'Livestock, farm animals', 103: 'Duck',
  105: 'Goose', 110: 'Pigeon, dove', 122: 'Insect', 500: 'Inside, small room'
};

// -------------------------- DERIVED TABLES -------------------------
const CLASS_TO_ANIMAL = {};   // classIndex -> animal key
const CLASS_NAMES = { ...EXTRA_CLASS_NAMES };
Object.keys(ANIMALS).forEach(key => {
  Object.keys(ANIMALS[key].classes).forEach(idx => {
    CLASS_TO_ANIMAL[idx] = key;
    CLASS_NAMES[idx] = ANIMALS[key].classes[idx];
  });
});

// ------------------------------ STATE ------------------------------
let model = null;
let audioCtx = null;
let started = false;
let inferBusy = false;

const ringBuffer = new Float32Array(INPUT_SAMPLES); // rolling 16 kHz mono audio
let ringFilled = 0;

let scoreHistory = [];        // rolling window of per-animal smoothed scores
let consecutiveLabel = null;
let consecutiveCount = 0;
let currentStable = null;
let lastEventTime = 0;
const counts = {};
Object.keys(ANIMALS).forEach(k => { counts[k] = 0; });

// ------------------------------ INIT UI ----------------------------
(function buildStaticUI() {
  // chips of recognizable animals
  const chips = document.getElementById('animal_chips');
  Object.keys(ANIMALS).forEach(key => {
    const a = ANIMALS[key];
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.id = 'chip_' + key;
    chip.textContent = a.emoji + ' ' + a.name;
    chips.appendChild(chip);
  });
  // counter tiles
  const counters = document.getElementById('counters');
  Object.keys(ANIMALS).forEach(key => {
    const a = ANIMALS[key];
    const tile = document.createElement('div');
    tile.className = 'counter';
    tile.innerHTML = '<span class="counter-emoji">' + a.emoji + '</span>' +
                     '<span>' + a.name + '</span>' +
                     '<strong id="count_' + key + '">0</strong>';
    counters.appendChild(tile);
  });
})();

// --------------------------- MODEL LOADER --------------------------
async function loadYamnet() {
  let lastErr = null;
  for (const src of MODEL_SOURCES) {
    try {
      const m = await tf.loadGraphModel(src.url, src.options);
      // Warm-up run so the first real prediction is fast
      tf.tidy(() => { m.predict(tf.zeros([INPUT_SAMPLES])); });
      console.log('YAMNet loaded from: ' + src.url);
      return m;
    } catch (err) {
      console.warn('Failed to load YAMNet from ' + src.url, err);
      lastErr = err;
    }
  }
  throw lastErr || new Error('All model sources failed');
}

// ------------------------------ START ------------------------------
async function startClassification() {
  if (started) return;
  const btn = document.getElementById('start_btn');
  btn.disabled = true;

  try {
    setStatus('Loading Google YAMNet model (~4 MB)…');
    model = await loadYamnet();

    setStatus('Requesting microphone…');
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: true }
    });

    audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
    const source = audioCtx.createMediaStreamSource(stream);
    const processor = audioCtx.createScriptProcessor(4096, 1, 1);

    source.connect(processor);
    processor.connect(audioCtx.destination); // required by some browsers

    processor.onaudioprocess = (e) => {
      const input = e.inputBuffer.getChannelData(0);
      pushAudio(input, e.inputBuffer.sampleRate);
      updateMicLevel(input);
    };

    started = true;
    btn.classList.add('listening');
    btn.innerHTML = '<span class="btn-icon">🔊</span> Listening…';
    btn.disabled = false;
    setStatus('✅ Listening… play any animal sound!');

    setInterval(runInference, INFER_INTERVAL_MS);
  } catch (err) {
    console.error(err);
    btn.disabled = false;
    if (err.name === 'NotAllowedError' || err.name === 'SecurityError') {
      setStatus('❌ Microphone access denied. Please allow the mic and reload.');
    } else {
      setStatus('❌ Could not load YAMNet (' + (err.message || err) +
                '). Check your internet connection / ad-blocker and reload.');
    }
  }
}

// -------------------------- AUDIO BUFFERING ------------------------
function pushAudio(chunk, srcRate) {
  // Resample (linear) to 16 kHz if the context didn't honor sampleRate
  let data = chunk;
  if (srcRate !== SAMPLE_RATE) {
    const ratio = srcRate / SAMPLE_RATE;
    const outLen = Math.floor(chunk.length / ratio);
    data = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const pos = i * ratio;
      const i0 = Math.floor(pos);
      const i1 = Math.min(i0 + 1, chunk.length - 1);
      data[i] = chunk[i0] + (chunk[i1] - chunk[i0]) * (pos - i0);
    }
  }
  // Slide the ring buffer left and append the new chunk
  if (data.length >= INPUT_SAMPLES) {
    ringBuffer.set(data.subarray(data.length - INPUT_SAMPLES));
  } else {
    ringBuffer.copyWithin(0, data.length);
    ringBuffer.set(data, INPUT_SAMPLES - data.length);
  }
  ringFilled = Math.min(INPUT_SAMPLES, ringFilled + data.length);
}

// ----------------------------- INFERENCE ---------------------------
async function runInference() {
  if (!model || inferBusy || ringFilled < INPUT_SAMPLES) return;
  inferBusy = true;
  try {
    const scores = tf.tidy(() => {
      const waveform = tf.tensor1d(ringBuffer);
      let out = model.predict(waveform);         // [scores, embeddings, spectrogram]
      if (!Array.isArray(out)) out = [out];
      // Pick the output whose last dimension is 521 (the class scores)
      let s = out[0];
      for (const t of out) {
        if (t.shape[t.shape.length - 1] === 521) { s = t; break; }
      }
      return s.mean(0);                          // average over frames -> [521]
    });
    const scoreArr = await scores.data();
    scores.dispose();
    processScores(scoreArr);
  } catch (err) {
    console.error('Inference error:', err);
  } finally {
    inferBusy = false;
  }
}

// -------------------------- SCORE PROCESSING -----------------------
function processScores(scoreArr) {
  // 1. Live bars: top-5 raw classes for transparency
  renderLiveBars(scoreArr);

  // 2. Aggregate 521 classes into per-animal scores (max of its classes)
  const frame = {};
  Object.keys(ANIMALS).forEach(key => {
    let best = 0;
    Object.keys(ANIMALS[key].classes).forEach(idx => {
      if (scoreArr[idx] > best) best = scoreArr[idx];
    });
    frame[key] = best;
  });

  // 3. Temporal smoothing across the rolling window
  scoreHistory.push(frame);
  if (scoreHistory.length > SMOOTHING_WINDOW) scoreHistory.shift();
  const avg = {};
  Object.keys(ANIMALS).forEach(key => {
    let sum = 0;
    scoreHistory.forEach(f => { sum += f[key]; });
    avg[key] = sum / scoreHistory.length;
  });

  // 4. Pick the smoothed winner
  let bestKey = null, bestScore = 0;
  Object.keys(avg).forEach(key => {
    if (avg[key] > bestScore) { bestScore = avg[key]; bestKey = key; }
  });

  highlightChip(bestScore >= CONFIDENCE_THRESHOLD ? bestKey : null);

  // 5. Threshold + consecutive-frame lock
  const passes = bestScore >= CONFIDENCE_THRESHOLD;
  if (passes) {
    if (bestKey === consecutiveLabel) consecutiveCount++;
    else { consecutiveLabel = bestKey; consecutiveCount = 1; }
  } else {
    consecutiveLabel = null;
    consecutiveCount = 0;
  }

  if (consecutiveCount >= CONSECUTIVE_NEEDED) {
    acceptDetection(bestKey, bestScore);
  } else if (bestScore < CONFIDENCE_THRESHOLD * 0.4) {
    resetToIdle();
  }

  updateConfidenceBar(bestScore, passes);
}

// ------------------------- ACCEPT A DETECTION ----------------------
function acceptDetection(key, score) {
  const a = ANIMALS[key];
  const pct = Math.round(score * 100);

  document.getElementById('result_label').innerHTML =
    'Detected voice is of — <strong>' + a.emoji + ' ' + a.name + '</strong> (' + pct + '% confidence)';

  const img = document.getElementById('animal_image');
  const emoji = document.getElementById('animal_emoji');
  if (a.img) {
    img.src = a.img;
    img.classList.remove('hidden');
    emoji.classList.add('hidden');
  } else {
    img.classList.add('hidden');
    emoji.textContent = a.emoji;
    emoji.classList.remove('hidden');
  }

  const now = Date.now();
  if (key !== currentStable || (now - lastEventTime) > EVENT_COOLDOWN_MS) {
    counts[key]++;
    document.getElementById('count_' + key).textContent = counts[key];
    addLogEntry(a.emoji + ' ' + a.name, pct);
    lastEventTime = now;
  }
  currentStable = key;
}

function resetToIdle() {
  if (currentStable !== null) {
    currentStable = null;
    document.getElementById('result_label').innerHTML = 'I can hear — nothing yet';
    const img = document.getElementById('animal_image');
    img.src = 'listen.gif';
    img.classList.remove('hidden');
    document.getElementById('animal_emoji').classList.add('hidden');
  }
}

// ----------------------------- UI HELPERS --------------------------
function setStatus(msg) {
  document.getElementById('status_text').textContent = msg;
}

function updateMicLevel(samples) {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  const rms = Math.sqrt(sum / samples.length);
  const pct = Math.min(100, Math.round(rms * 400));
  document.getElementById('mic_level').style.width = pct + '%';
}

function renderLiveBars(scoreArr) {
  // top 5 classes overall
  const top = Array.from(scoreArr.keys())
    .sort((a, b) => scoreArr[b] - scoreArr[a])
    .slice(0, 5);

  const wrap = document.getElementById('live_bars');
  wrap.innerHTML = '';
  top.forEach(idx => {
    const pct = Math.round(scoreArr[idx] * 100);
    const name = CLASS_NAMES[idx] || ('Class #' + idx);
    const isAnimal = CLASS_TO_ANIMAL.hasOwnProperty(idx);
    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML =
      '<span class="bar-label' + (isAnimal ? ' animal' : '') + '">' + name + '</span>' +
      '<div class="bar-track"><div class="bar-fill' + (isAnimal ? ' animal' : '') +
      '" style="width:' + pct + '%"></div></div>' +
      '<span class="bar-pct">' + pct + '%</span>';
    wrap.appendChild(row);
  });
}

function updateConfidenceBar(score, passes) {
  const pct = Math.min(100, Math.round(score * 100));
  const bar = document.getElementById('confidence_bar');
  bar.style.width = pct + '%';
  bar.className = 'confidence-fill ' + (passes ? 'good' : 'low');
  document.getElementById('confidence_text').textContent = pct + '%';
}

function highlightChip(activeKey) {
  Object.keys(ANIMALS).forEach(key => {
    const chip = document.getElementById('chip_' + key);
    if (chip) chip.classList.toggle('active', key === activeKey);
  });
}

function addLogEntry(name, pct) {
  const log = document.getElementById('detection_log');
  const placeholder = log.querySelector('.muted');
  if (placeholder) placeholder.remove();

  const li = document.createElement('li');
  li.textContent = new Date().toLocaleTimeString() + ' — ' + name + ' (' + pct + '%)';
  log.prepend(li);
  while (log.children.length > 8) log.removeChild(log.lastChild);
}
