function toggleNotebookFullscreen(button) {
  var container = button.closest('.iframe-container');
  container.classList.toggle('expanded');
  if (container.classList.contains('expanded')) {
    document.body.style.overflow = 'hidden';
    button.textContent = 'Exit Fullscreen';
  } else {
    document.body.style.overflow = '';
    button.textContent = 'Expand';
  }
}
