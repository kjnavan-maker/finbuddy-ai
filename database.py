"""Database layer for FinBuddy AI."""

import sqlite3
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

DB_NAME = "finance.db"


def get_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS expenses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        amount REAL NOT NULL,
        category TEXT NOT NULL,
        description TEXT,
        expense_date TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS budgets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        month TEXT NOT NULL,
        amount REAL NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE(user_id, month),
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    # GLOBAL AI KNOWLEDGE BASE
    cur.execute("""
    CREATE TABLE IF NOT EXISTS learned_responses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT UNIQUE NOT NULL,
        answer TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS finance_tips (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tip TEXT NOT NULL,
        category TEXT NOT NULL
    )
    """)

    # USER-SPECIFIC FINANCE PROFILE
    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_finance_profile (
        user_id INTEGER PRIMARY KEY,
        salary REAL DEFAULT 0,
        savings REAL DEFAULT 0,
        financial_goal TEXT,
        goal_amount REAL DEFAULT 0,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    seed_tips(cur)
    conn.commit()
    conn.close()


def seed_tips(cur):
    cur.execute("SELECT COUNT(*) AS total FROM finance_tips")
    if cur.fetchone()["total"] > 0:
        return

    tips = [
        ("Use the 50/30/20 rule: needs 50%, wants 30%, savings 20%.", "saving"),
        ("Build an emergency fund covering at least 3 months of essential expenses.", "saving"),
        ("Track small daily purchases; they often become large monthly leaks.", "expense"),
        ("Pay high-interest debts first to reduce total interest cost.", "debt"),
        ("Set a realistic budget and review it every month.", "budget"),
        ("Avoid lifestyle inflation when income increases; increase savings too.", "saving"),
        ("Compare subscriptions and cancel services you rarely use.", "expense")
    ]

    cur.executemany(
        "INSERT INTO finance_tips (tip, category) VALUES (?, ?)",
        tips
    )


def create_user(username, password):
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, generate_password_hash(password), datetime.now().isoformat())
        )
        conn.commit()
        return True, "Registration successful. Please log in."
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    finally:
        conn.close()


def validate_user(username, password):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cur.fetchone()
    conn.close()

    if user and check_password_hash(user["password_hash"], password):
        return dict(user)

    return None


def add_expense(user_id, amount, category, description=""):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO expenses
        (user_id, amount, category, description, expense_date, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        float(amount),
        category.lower().strip(),
        description,
        datetime.now().date().isoformat(),
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()


def set_budget(user_id, amount, month=None):
    month = month or datetime.now().strftime("%Y-%m")

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO budgets (user_id, month, amount, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(user_id, month)
        DO UPDATE SET amount=excluded.amount
    """, (
        user_id,
        month,
        float(amount),
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()


def get_budget(user_id, month=None):
    month = month or datetime.now().strftime("%Y-%m")

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT amount FROM budgets WHERE user_id = ? AND month = ?",
        (user_id, month)
    )

    row = cur.fetchone()
    conn.close()

    return row["amount"] if row else 0


def get_monthly_expenses(user_id, month=None):
    month = month or datetime.now().strftime("%Y-%m")

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT category, SUM(amount) AS total
        FROM expenses
        WHERE user_id = ?
        AND substr(expense_date, 1, 7) = ?
        GROUP BY category
        ORDER BY total DESC
    """, (user_id, month))

    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    return rows


def get_recent_expenses(user_id, limit=10):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT amount, category, description, expense_date
        FROM expenses
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT ?
    """, (user_id, limit))

    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    return rows


def save_learned_response(user_id, question, answer):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO learned_responses (question, answer, created_at)
        VALUES (?, ?, ?)
        ON CONFLICT(question)
        DO UPDATE SET answer=excluded.answer
    """, (
        question.lower().strip(),
        answer.strip(),
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()


def get_learned_responses(user_id=None):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT question, answer FROM learned_responses")

    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    return rows


def save_user_profile(user_id, salary=None, savings=None, financial_goal=None, goal_amount=None):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT * FROM user_finance_profile WHERE user_id = ?", (user_id,))
    existing = cur.fetchone()

    if existing:
        cur.execute("""
            UPDATE user_finance_profile
            SET salary = COALESCE(?, salary),
                savings = COALESCE(?, savings),
                financial_goal = COALESCE(?, financial_goal),
                goal_amount = COALESCE(?, goal_amount),
                updated_at = ?
            WHERE user_id = ?
        """, (
            salary,
            savings,
            financial_goal,
            goal_amount,
            datetime.now().isoformat(),
            user_id
        ))
    else:
        cur.execute("""
            INSERT INTO user_finance_profile
            (user_id, salary, savings, financial_goal, goal_amount, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            salary or 0,
            savings or 0,
            financial_goal,
            goal_amount or 0,
            datetime.now().isoformat()
        ))

    conn.commit()
    conn.close()


def get_user_profile(user_id):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT * FROM user_finance_profile WHERE user_id = ?", (user_id,))
    row = cur.fetchone()

    conn.close()

    if row:
        return dict(row)

    return {
        "salary": 0,
        "savings": 0,
        "financial_goal": None,
        "goal_amount": 0
    }


def get_random_tip():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT tip FROM finance_tips ORDER BY RANDOM() LIMIT 1")

    row = cur.fetchone()
    conn.close()

    return row["tip"] if row else "Track your expenses daily to stay financially aware."