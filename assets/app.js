const ui = new WebUI();
const MAX_HISTORY = 8;
let history = [];
let overlayTimer;
let cameraEnrollmentActive = false;

const $ = id => document.getElementById(id);
const errorContainer = $('error-container');
const connectionBadge = $('connectionBadge');
const enrollmentForm = $('enrollmentForm');
const enrollmentStatus = $('enrollmentStatus');
const enrollButton = $('enrollButton');
const cancelEnrollmentButton = $('cancelEnrollmentButton');

ui.on_connect(() => {
  errorContainer.hidden = true;
  connectionBadge.textContent = 'Online';
  connectionBadge.className = 'status-pill online';
  ui.send_message('list_people', {});
});

ui.on_disconnect(() => {
  errorContainer.textContent = 'Connection to the UNO Q was lost. Check the board and network.';
  errorContainer.hidden = false;
  connectionBadge.textContent = 'Offline';
  connectionBadge.className = 'status-pill offline';
});

ui.on_message('recognition', message => {
  const faces = message.faces || [];
  if (!faces.length) return;
  const best = [...faces].sort((a, b) => b.similarity - a.similarity)[0];
  showRecognition(best);
  renderFaceLabels(faces, message.frame_width, message.frame_height);
  history.unshift(...faces);
  history = history.slice(0, MAX_HISTORY);
  renderHistory();
});

ui.on_message('people', message => renderPeople(message.people || []));
ui.on_message('app_error', message => showError(message.message || 'Unexpected application error'));
ui.on_message('enrollment_result', message => {
  cameraEnrollmentActive = false;
  enrollButton.disabled = false;
  cancelEnrollmentButton.hidden = true;
  if (message.ok) {
    enrollmentStatus.textContent = `${message.name} saved with ${message.sample_count} strong, varied face sample(s).`;
    enrollmentStatus.className = 'form-status success';
    enrollmentForm.reset();
  } else if (message.cancelled) {
    enrollmentStatus.textContent = message.message;
    enrollmentStatus.className = 'form-status';
  } else {
    enrollmentStatus.textContent = message.message || 'Enrollment failed.';
    enrollmentStatus.className = 'form-status failure';
  }
});
ui.on_message('enrollment_progress', message => {
  enrollmentStatus.textContent = `Captured ${message.accepted} of ${message.total}.${message.rejected_reason ? ` ${message.rejected_reason}.` : ' Good—change your head angle slightly.'}`;
});

$('confidenceSlider').addEventListener('input', event => {
  const value = Number(event.target.value);
  $('confidenceValueDisplay').textContent = value.toFixed(2);
  ui.send_message('override_th', value);
});

enrollmentForm.addEventListener('submit', event => {
  event.preventDefault();
  if (cameraEnrollmentActive) return;
  cameraEnrollmentActive = true;
  enrollButton.disabled = true;
  cancelEnrollmentButton.hidden = false;
  enrollmentStatus.textContent = 'Starting the camera. Keep one face centered and well lit…';
  enrollmentStatus.className = 'form-status';
  ui.send_message('camera_enrollment_start', {
    name: $('personName').value,
    sample_count: Number($('sampleCount').value),
  });
});

cancelEnrollmentButton.addEventListener('click', () => {
  if (!cameraEnrollmentActive) return;
  ui.send_message('camera_enrollment_cancel', {});
});

function showRecognition(face) {
  const known = face.name !== 'Unknown';
  const panel = $('recognitionResult');
  panel.className = `recognition-result ${known ? 'known' : 'unknown'}`;
  panel.innerHTML = `<span class="result-dot"></span><div><strong>${escapeHtml(face.name)}</strong><small>Similarity ${(face.similarity * 100).toFixed(1)}% · detector ${(face.detection_confidence * 100).toFixed(1)}%</small></div>`;
}

function renderFaceLabels(faces, frameWidth, frameHeight) {
  const overlay = $('faceOverlay');
  if (!frameWidth || !frameHeight) return;
  overlay.innerHTML = '';
  faces.forEach(face => {
    const [x1, y1, x2, y2] = face.bounding_box || [];
    if (![x1, y1, x2, y2].every(Number.isFinite)) return;
    const label = document.createElement('div');
    label.className = `face-name-label ${face.name === 'Unknown' ? 'unknown' : 'known'}`;
    label.textContent = `${face.name} ${(face.similarity * 100).toFixed(0)}%`;
    label.style.left = `${Math.max(0, Math.min(100, x1 / frameWidth * 100))}%`;
    label.style.top = `${Math.max(0, Math.min(100, y1 / frameHeight * 100))}%`;
    label.style.maxWidth = `${Math.max(15, (x2 - x1) / frameWidth * 100)}%`;
    overlay.appendChild(label);
  });
  clearTimeout(overlayTimer);
  overlayTimer = setTimeout(() => { overlay.innerHTML = ''; }, 900);
}

function renderPeople(people) {
  $('peopleCount').textContent = people.length;
  const list = $('peopleList');
  if (!people.length) {
    list.innerHTML = '<p class="empty-state">No one enrolled yet.</p>';
    return;
  }
  list.innerHTML = people.map(person => `
    <div class="person-row">
      <span class="avatar">${escapeHtml(person.name.charAt(0).toUpperCase())}</span>
      <span><strong>${escapeHtml(person.name)}</strong><small>${person.sample_count} face template(s)</small></span>
      <button class="delete-person" data-id="${person.id}" data-name="${escapeHtml(person.name)}" type="button">Remove</button>
    </div>`).join('');
  list.querySelectorAll('.delete-person').forEach(button => button.addEventListener('click', () => {
    if (window.confirm(`Remove ${button.dataset.name} and all stored face photos?`)) {
      ui.send_message('delete_person', {person_id: Number(button.dataset.id)});
    }
  }));
}

function renderHistory() {
  $('recentDetections').innerHTML = history.map(face => `
    <div class="history-row">
      <span class="history-status ${face.name === 'Unknown' ? 'unknown' : 'known'}"></span>
      <span><strong>${escapeHtml(face.name)}</strong><small>${new Date(face.timestamp).toLocaleTimeString()} · ${(face.similarity * 100).toFixed(1)}%</small></span>
    </div>`).join('');
}

function setEnrollmentError(message) {
  enrollmentStatus.textContent = message;
  enrollmentStatus.className = 'form-status failure';
}

function showError(message) {
  errorContainer.textContent = message;
  errorContainer.hidden = false;
}

function escapeHtml(value) {
  const element = document.createElement('span');
  element.textContent = String(value);
  return element.innerHTML;
}
