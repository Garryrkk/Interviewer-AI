/* ===============================
   Global state
================================= */
let isInvisible = false;
let isPrepVoice = false;
let isProcessing = false;

/* ===============================
   Feature Navigation
================================= */
export function showFeature(featureId, e) {
  // Hide all panels
  document.querySelectorAll('.feature-panel').forEach(panel => {
    panel.classList.remove('active');
  });

  // Remove active class from nav links + feature cards
  document.querySelectorAll('.nav-link.active, .feature-card.active')
          .forEach(el => el.classList.remove('active'));

  // Show selected panel
  document.getElementById(featureId)?.classList.add('active');

  // Add active class to clicked nav link (if event provided)
  if (e) {
    const link = e.target.closest('.nav-link');
    if (link) link.classList.add('active');
  }

  // Highlight feature card if it exists
  const featureCard = document.querySelector(
    `[data-feature="${featureId}"].feature-card`
  );
  featureCard?.classList.add('active');
}

/* ===============================
   UI Toggles
================================= */
export function toggleSidebar() {
  document.getElementById('sidebar')?.classList.toggle('collapsed');
}

export function toggleInvisibility() {
  isInvisible = !isInvisible;
  const icon  = document.getElementById('invisibilityIcon');
  const title = document.getElementById('invisibilityTitle');
  const desc  = document.getElementById('invisibilityDesc');
  const btn   = document.getElementById('invisibilityBtn');

  if (!icon || !title || !desc || !btn) return;

  if (isInvisible) {
    icon.innerHTML = '<i class="fas fa-eye-slash"></i>';
    icon.style.color = '#10b981';
    title.textContent = 'Invisibility Mode Active';
    desc.textContent  = 'The AI assistant is running invisibly in the background, monitoring and analyzing without detection.';
    btn.innerHTML     = '<i class="fas fa-eye"></i> Deactivate Invisibility';
    btn.className     = 'btn btn-danger';
  } else {
    icon.innerHTML = '<i class="fas fa-eye"></i>';
    icon.style.color = '#8F74D4';
    title.textContent = 'Invisibility Mode Inactive';
    desc.textContent  = 'Activate invisible mode for stealth operation during sensitive conversations or interviews. The AI will monitor and analyze without detection.';
    btn.innerHTML     = '<i class="fas fa-eye-slash"></i> Activate Invisibility';
    btn.className     = 'btn';
  }
}

export function togglePrepVoice(e) {
  isPrepVoice = !isPrepVoice;
  const indicator = document.getElementById('prepIndicator');
  const btn = e?.target;

  if (!indicator || !btn) return;

  indicator.classList.toggle('active', isPrepVoice);
  btn.textContent = isPrepVoice ? 'Stop Prep' : 'Start Prep';
  btn.className   = isPrepVoice ? 'btn btn-danger' : 'btn';

  updateVoiceStatus();
}

export function toggleProcessing(e) {
  isProcessing = !isProcessing;
  const indicator = document.getElementById('processIndicator');
  const btn = e?.target;

  if (!indicator || !btn) return;

  indicator.classList.toggle('active', isProcessing);
  btn.textContent = isProcessing ? 'Stop' : 'Start';
  btn.className   = 'btn btn-danger';

  updateVoiceStatus();
}

export function recognizeVoice() {
  const indicator = document.getElementById('recognitionIndicator');
  if (!indicator) return;

  indicator.classList.add('active');
  setTimeout(() => indicator.classList.remove('active'), 2000);
}

function updateVoiceStatus() {
  const status = document.getElementById('voiceStatus');
  if (!status) return;

  const active = isPrepVoice || isProcessing;
  status.textContent = active ? 'Active' : 'Standby';
  status.className   = active
    ? 'status-badge status-active'
    : 'status-badge status-inactive';
}

/* ===============================
   Quick Actions
================================= */
export const activateQuickResponse = () => showFeature('quickresponse');
export const generateSummary      = () => showFeature('summarization');
export const extractInsights      = () => showFeature('keyinsights');
export const showSettings         = () => alert('Settings panel would open here');
export const showProfile          = () => alert('Profile panel would open here');

/* ===============================
   Bootstrapping
================================= */
document.addEventListener('DOMContentLoaded', () => {
  // Wire nav links dynamically instead of inline onclick
  document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', e => {
      e.preventDefault();
      const featureId = link.dataset.feature;
      if (featureId) showFeature(featureId, e);
    });
  });
});
