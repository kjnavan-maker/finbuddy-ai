"""Flask backend for FinBuddy AI personal finance chatbot."""

import csv
import io
from datetime import datetime
from functools import wraps
from database import get_reminders

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    redirect,
    session,
    url_for,
    Response,
    flash
)

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from chatbot import FinBuddyAI
from database import (
    init_db,
    create_user,
    validate_user,
    get_monthly_expenses,
    get_recent_expenses,
    get_budget
)


app = Flask(__name__)
app.secret_key = "change-this-secret-key-for-production"

init_db()

try:
    import admin_training
except Exception as error:
    print("Admin training skipped:", error)

bot = FinBuddyAI()


def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return func(*args, **kwargs)

    return wrapper


@app.route("/")
@login_required
def index():

    current_user = session.get("username")

    reminders = get_reminders(session["user_id"])

    return render_template(
        "index.html",
        username=current_user,
        reminders=reminders
    )


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if not username or not password:
            flash("Username and password are required.")
        else:
            ok, message = create_user(username, password)
            flash(message)

            if ok:
                return redirect(url_for("login"))

    return render_template(
        "index.html",
        auth_mode="register"
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        user = validate_user(username, password)

        if user:
            session["user_id"] = user["id"]
            session["username"] = user["username"]

            return redirect(url_for("index"))

        flash("Invalid username or password.")

    return render_template(
        "index.html",
        auth_mode="login"
    )


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json() or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({
            "reply": "Please enter your financial question or expense details 😊",
            "intent": "empty"
        })

    response = bot.reply(
        session["user_id"],
        message
    )

    return jsonify(response)


@app.route("/api/summary")
@login_required
def summary():
    rows = get_monthly_expenses(session["user_id"])
    total = sum(row["total"] for row in rows)
    budget = get_budget(session["user_id"])

    return jsonify({
        "categories": [row["category"] for row in rows],
        "totals": [row["total"] for row in rows],
        "total_spent": total,
        "budget": budget,
        "remaining": budget - total if budget else 0,
        "recent": get_recent_expenses(session["user_id"], 8)
    })


@app.route("/export/csv")
@login_required
def export_csv():
    rows = get_recent_expenses(session["user_id"], 1000)

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "Date",
        "Category",
        "Amount",
        "Description"
    ])

    for row in rows:
        writer.writerow([
            row["expense_date"],
            row["category"],
            row["amount"],
            row["description"]
        ])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=finbuddy_expenses.csv"
        }
    )


@app.route("/export/pdf")
@login_required
def export_pdf():
    rows = get_recent_expenses(session["user_id"], 1000)

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)

    pdf.setTitle("FinBuddy AI Expense Report")

    y = 750

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, y, "FinBuddy AI - Expense Report")

    y -= 30

    pdf.setFont("Helvetica", 10)
    pdf.drawString(
        50,
        y,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )

    y -= 30

    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(50, y, "Date")
    pdf.drawString(150, y, "Category")
    pdf.drawString(280, y, "Amount")
    pdf.drawString(370, y, "Description")

    y -= 18

    pdf.setFont("Helvetica", 9)

    for row in rows:
        if y < 60:
            pdf.showPage()
            y = 750
            pdf.setFont("Helvetica", 9)

        pdf.drawString(50, y, str(row["expense_date"]))
        pdf.drawString(150, y, str(row["category"])[:18])
        pdf.drawString(280, y, f"{row['amount']:.2f}")
        pdf.drawString(370, y, str(row["description"])[:40])

        y -= 16

    pdf.save()
    buffer.seek(0)

    return Response(
        buffer.getvalue(),
        mimetype="application/pdf",
        headers={
            "Content-Disposition": "attachment; filename=finbuddy_expenses.pdf"
        }
    )

@app.route("/api/reminders")
@login_required
def reminders_api():
    reminders = get_reminders(session["user_id"])

    formatted = []
    for r in reminders:
        formatted.append({
            "bill_name": r["bill_name"],
            "due_day": r["due_day"]
        })

    return jsonify(formatted)

if __name__ == "__main__":
    app.run(debug=True)