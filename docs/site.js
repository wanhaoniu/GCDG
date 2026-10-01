'use strict';
function addRows(id, rows) {
  const body = document.getElementById(id);
  for (const row of rows) {
    const tr = document.createElement('tr');
    if (/GCDG|Ours/.test(row[0])) tr.className = 'ours';
    row.forEach((value, index) => {
      const cell = document.createElement(index ? 'td' : 'th');
      if (!index) cell.scope = 'row';
      cell.textContent = value; tr.appendChild(cell);
    });
    body.appendChild(tr);
  }
}
addRows('prediction-rows', window.GCDG_RESULTS.prediction);
addRows('ordered-rows', window.GCDG_RESULTS.ordered);
const tabs = Array.from(document.querySelectorAll('[role=tab]'));
function selectTab(tab) {
  for (const item of tabs) {
    const active = item === tab;
    item.setAttribute('aria-selected', String(active));
    item.tabIndex = active ? 0 : -1;
    document.getElementById(item.getAttribute('aria-controls')).hidden = !active;
  }
}
tabs.forEach((tab, index) => {
  tab.addEventListener('click', () => selectTab(tab));
  tab.addEventListener('keydown', event => {
    let next;
    if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
    if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = tabs.length - 1;
    if (next !== undefined) { event.preventDefault(); selectTab(tabs[next]); tabs[next].focus(); }
  });
});
