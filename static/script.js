let expenseChart = null;

function addMessage(text, sender) {
    const chatBox = document.getElementById('chatBox');
    if (!chatBox) return;
    const div = document.createElement('div');
    div.className = `message ${sender}`;
    div.innerHTML = text;
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
}

async function sendMessage() {
    const input = document.getElementById('userInput');
    const typing = document.getElementById('typing');
    const message = input.value.trim();
    if (!message) return;
    addMessage(message, 'user');
    input.value = '';
    typing.classList.remove('hidden');

    const response = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
    });
    const data = await response.json();
    setTimeout(() => {
        typing.classList.add('hidden');
        addMessage(data.reply, 'bot');
        loadSummary();
    }, 450);
}

async function loadSummary() {
    const totalSpent = document.getElementById('totalSpent');
    if (!totalSpent) return;
    const res = await fetch('/api/summary');
    const data = await res.json();
    totalSpent.textContent = Number(data.total_spent).toFixed(2);
    document.getElementById('budgetAmount').textContent = Number(data.budget).toFixed(2);
    document.getElementById('remainingAmount').textContent = Number(data.remaining).toFixed(2);

    const recent = document.getElementById('recentExpenses');
    recent.innerHTML = data.recent.length ? '' : '<p class="muted">No expenses yet.</p>';
    data.recent.forEach(item => {
        const row = document.createElement('div');
        row.className = 'expense-row';
        row.innerHTML = `<span>${item.category}<br><small>${item.expense_date}</small></span><strong>${Number(item.amount).toFixed(2)}</strong>`;
        recent.appendChild(row);
    });

    const canvas = document.getElementById('expenseChart');
    if (!canvas) return;
    if (expenseChart) expenseChart.destroy();
    expenseChart = new Chart(canvas, {
        type: 'doughnut',
        data: {
            labels: data.categories,
            datasets: [{ data: data.totals }]
        },
        options: { responsive: true, plugins: { legend: { position: 'bottom' } } }
    });
}

function setupVoiceInput() {
    const voiceBtn = document.getElementById('voiceBtn');
    const input = document.getElementById('userInput');
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!voiceBtn || !SpeechRecognition) return;
    const recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.onresult = event => { input.value = event.results[0][0].transcript; };
    voiceBtn.onclick = () => recognition.start();
}

document.addEventListener('DOMContentLoaded', () => {
    const sendBtn = document.getElementById('sendBtn');
    const input = document.getElementById('userInput');
    const darkToggle = document.getElementById('darkToggle');
    if (sendBtn) sendBtn.onclick = sendMessage;
    if (input) input.addEventListener('keydown', e => { if (e.key === 'Enter') sendMessage(); });
    if (darkToggle) darkToggle.onclick = () => document.body.classList.toggle('dark');
    setupVoiceInput();
    loadSummary();
});
