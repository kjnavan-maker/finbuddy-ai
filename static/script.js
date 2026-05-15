// ===============================
// FINBUDDY AI - FINAL SCRIPT.JS
// ===============================

let expenseChart = null;

const sendBtn = document.getElementById("sendBtn");
const userInput = document.getElementById("userInput");
const chatBox = document.getElementById("chatBox");
const typing = document.getElementById("typing");
const themeBtn = document.getElementById("darkToggle");

// ===============================
// DARK / LIGHT MODE
// ===============================
function updateThemeText() {

    const themeIcon = document.getElementById("themeIcon");
    const themeText = document.getElementById("themeText");

    if (!themeIcon || !themeText) return;

    if (document.body.classList.contains("dark-theme")) {

        themeIcon.className = "fa-solid fa-sun nav-icon";
        themeText.textContent = "Light Mode";

    } else {

        themeIcon.className = "fa-solid fa-moon nav-icon";
        themeText.textContent = "Dark Mode";

    }
}

if (themeBtn) {
    themeBtn.addEventListener("click", () => {
        document.body.classList.toggle("dark-theme");

        localStorage.setItem(
            "theme",
            document.body.classList.contains("dark-theme") ? "light" : "dark"
        );

        updateThemeText();
    });
}

// ===============================
// SIDEBAR COLLAPSE
// ===============================
function setupSidebarToggle() {
    const menuBtn = document.querySelector(".menu-btn");
    const appLayout = document.querySelector(".app-layout");

    if (!menuBtn || !appLayout) return;

    menuBtn.addEventListener("click", function () {
        appLayout.classList.toggle("sidebar-collapsed");
    });
}

// ===============================
// CHAT MESSAGE DISPLAY
// ===============================
function addMessage(text, sender) {
    if (!chatBox) return;

    const msg = document.createElement("div");
    msg.className = `message ${sender}`;
    msg.innerHTML = text.replace(/\n/g, "<br>");

    chatBox.appendChild(msg);
    chatBox.scrollTop = chatBox.scrollHeight;
}

// ===============================
// SEND MESSAGE
// ===============================
async function sendMessage() {
    if (!userInput) return;

    const message = userInput.value.trim();
    if (!message) return;

    addMessage(message, "user");
    userInput.value = "";

    if (typing) typing.classList.remove("hidden");

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message: message })
        });

        const data = await response.json();

        if (typing) typing.classList.add("hidden");

        addMessage(
            data.reply || data.response || "Sorry, I could not understand.",
            "bot"
        );

        loadSummary();
        loadRemindersFromAPI();

    } catch (error) {
        if (typing) typing.classList.add("hidden");
        addMessage("Server error. Please try again.", "bot");
        console.error("Chat Error:", error);
    }
}

if (sendBtn) {
    sendBtn.addEventListener("click", sendMessage);
}

if (userInput) {
    userInput.addEventListener("keydown", function (event) {
        if (event.key === "Enter") {
            sendMessage();
        }
    });
}

// ===============================
// QUICK MESSAGE
// ===============================
function quickMessage(text) {
    if (userInput && sendBtn) {
        userInput.value = text;
        sendBtn.click();
    }
}

// ===============================
// VOICE INPUT
// ===============================
const voiceBtn = document.getElementById("voiceBtn");

if (voiceBtn) {
    voiceBtn.addEventListener("click", () => {
        const SpeechRecognition =
            window.SpeechRecognition || window.webkitSpeechRecognition;

        if (!SpeechRecognition) {
            alert("Voice recognition is not supported in this browser.");
            return;
        }

        const recognition = new SpeechRecognition();
        recognition.lang = "en-US";
        recognition.start();

        recognition.onresult = function (event) {
            userInput.value = event.results[0][0].transcript;
        };
    });
}

// ===============================
// LOAD DASHBOARD SUMMARY
// ===============================
async function loadSummary() {
    try {
        const response = await fetch("/api/summary");
        const data = await response.json();

        const totalSpent = document.getElementById("totalSpent");
        const budgetAmount = document.getElementById("budgetAmount");
        const remainingAmount = document.getElementById("remainingAmount");

        if (totalSpent) {
            totalSpent.textContent = Number(data.total_spent || 0).toFixed(2);
        }

        if (budgetAmount) {
            budgetAmount.textContent = Number(data.budget || 0).toFixed(2);
        }

        if (remainingAmount) {
            remainingAmount.textContent = Number(data.remaining || 0).toFixed(2);
        }

        renderRecentExpenses(data.recent || []);

        const categoryData = {};
        const categories = data.categories || [];
        const totals = data.totals || [];

        categories.forEach((category, index) => {
            categoryData[category] = totals[index] || 0;
        });

        renderExpenseChart(categoryData);

    } catch (error) {
        console.error("Summary Error:", error);
    }
}

// ===============================
// RECENT EXPENSES
// ===============================
function renderRecentExpenses(expenses) {
    const recentExpenses = document.getElementById("recentExpenses");
    if (!recentExpenses) return;

    if (!expenses.length) {
        recentExpenses.innerHTML = `<p class="muted">No expenses yet.</p>`;
        return;
    }

    recentExpenses.innerHTML = expenses.map(expense => `
        <div class="expense-row">
            <div>
                <strong>${expense.category || "Expense"}</strong><br>
                <small>${expense.expense_date || expense.date || ""}</small>
            </div>
            <strong>${Number(expense.amount || 0).toFixed(2)}</strong>
        </div>
    `).join("");
}

// ===============================
// EXPENSE CHART
// ===============================
function renderExpenseChart(categories) {
    const canvas = document.getElementById("expenseChart");
    if (!canvas) return;

    const labels = Object.keys(categories);
    const values = Object.values(categories);

    if (expenseChart) {
        expenseChart.destroy();
    }

    expenseChart = new Chart(canvas, {
        type: "doughnut",
        data: {
            labels: labels.length ? labels : ["No Data"],
            datasets: [{
                data: values.length ? values : [1],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "bottom",
                    labels: {
                        color: "#94a3b8"
                    }
                }
            }
        }
    });
}

// ===============================
// DEFAULT REMINDERS
// ===============================
function getDefaultReminders() {
    return [
        { title: "EMI Payment", day: 15 },
        { title: "Electricity Bill", day: 15 },
        { title: "House Rent", day: 15 },
        { title: "Internet Bill", day: 19 },
        { title: "Water Bill", day: 25 }
    ];
}

// ===============================
// LOAD REMINDERS FROM BACKEND
// ===============================
async function loadRemindersFromAPI() {
    try {
        const response = await fetch("/api/reminders");
        const data = await response.json();

        const reminders = data.map(item => ({
            title: item.bill_name || item.title || "Bill Reminder",
            day: item.due_day || item.day || "15"
        }));

        renderReminders(reminders);

    } catch (error) {
        console.error("Reminder Error:", error);
        renderReminders(getDefaultReminders());
    }
}

// ===============================
// RENDER REMINDERS
// ===============================
function renderReminders(reminders) {
    const reminderListPopup = document.getElementById("reminderListPopup");
    const reminderCount = document.getElementById("reminderCount");

    const reminderData = reminders || [];

    const html = reminderData.map(item => `
        <div class="reminder-item">
            🔔 <strong>${item.title || "Bill Reminder"}</strong><br>
            <small>Every month on ${item.day || "15"}</small>
        </div>
    `).join("");

    if (reminderListPopup) {
        reminderListPopup.innerHTML =
        reminderData.length > 0
          ? html
            : `<p>No reminders found.</p>`;
    }

    if (reminderCount) {
        reminderCount.textContent = reminderData.length;
        reminderCount.style.display = reminderData.length > 0 ? "grid" : "none";
    }
}

// ===============================
// REMINDER POPUP
// ===============================
function setupReminderPopup() {
    const reminderBtn = document.getElementById("reminderBtn");
    const reminderPopup = document.getElementById("reminderPopup");

    if (!reminderBtn || !reminderPopup) return;

    reminderBtn.addEventListener("click", function (event) {
        event.stopPropagation();
        reminderPopup.classList.toggle("hidden");
    });

    reminderPopup.addEventListener("click", function (event) {
        event.stopPropagation();
    });

    document.addEventListener("click", function () {
        reminderPopup.classList.add("hidden");
    });
}

// ===============================
// INITIAL LOAD
// ===============================
document.addEventListener("DOMContentLoaded", function () {
    if (localStorage.getItem("theme") === "light") {
        document.body.classList.add("dark-theme");
    }

    updateThemeText();
    setupSidebarToggle();
    setupReminderPopup();
    loadSummary();
    loadRemindersFromAPI();
});