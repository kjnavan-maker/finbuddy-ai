"""Professional NLP, inference engine, finance logic, and self-learning for FinBuddy AI."""

import ast
import json
import operator
import os
import random
import re

import joblib
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from database import (
    add_expense,
    set_budget,
    get_budget,
    get_monthly_expenses,
    get_random_tip,
    save_learned_response,
    get_learned_responses,
    save_user_profile,
    get_user_profile,
    add_reminder,
    get_reminders,
)

from model_training import train_model


MODEL_FILE = "intent_model.joblib"
DATA_FILE = os.path.join("dataset", "intents.json")

lemmatizer = WordNetLemmatizer()


FINANCE_TERMS = {
    "budget": "A budget is a financial plan that helps you manage income, expenses, savings, and financial goals.",
    "savings": "Savings are money kept aside for future needs, emergencies, or important goals.",
    "saving": "Saving means keeping money aside before spending on wants.",
    "investment": "Investment means using money to buy assets that may grow in value over time.",
    "expense": "An expense is money spent on needs, wants, bills, or services.",
    "income": "Income is money you receive from salary, business, freelancing, or other sources.",
    "salary": "Salary is fixed income received from employment.",
    "debt": "Debt is borrowed money that must be repaid.",
    "loan": "A loan is borrowed money that must be repaid, usually with interest.",
    "emi": "EMI means Equated Monthly Installment, a fixed monthly loan repayment.",
    "interest": "Interest is the cost of borrowing money or the reward for saving money.",
    "inflation": "Inflation means prices increase over time, reducing purchasing power.",
    "emergency fund": "An emergency fund is money saved for unexpected situations.",
    "compound interest": "Compound interest means earning interest on both original money and previous interest.",
    "asset": "An asset is something valuable that you own.",
    "liability": "A liability is money you owe to others.",
    "net worth": "Net worth is total assets minus total liabilities.",
    "nlp": "NLP means Natural Language Processing, which helps computers understand human language.",
    "knowledge base": "A knowledge base stores facts, learned answers, finance tips, expenses, and budgets.",
    "inference engine": "An inference engine uses rules and user data to choose the best reply or financial advice."
}


def ensure_model():
    if not os.path.exists(MODEL_FILE):
        return train_model()
    return joblib.load(MODEL_FILE)


def safe_lemmatize(token):
    try:
        return lemmatizer.lemmatize(token)
    except LookupError:
        return token


def preprocess(text):
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return " ".join(safe_lemmatize(t) for t in tokens if t.isalnum())


class FinBuddyAI:
    def __init__(self):
        self.model = ensure_model()

        try:
            with open(DATA_FILE, "r", encoding="utf-8") as file:
                self.intents = json.load(file)["intents"]
        except Exception:
            self.intents = []

    def random_answer(self, answer):
        answers = [a.strip() for a in answer.split("|") if a.strip()]
        return random.choice(answers) if answers else answer

    def predict_intent(self, message):
        processed = preprocess(message)

        try:
            probabilities = self.model.predict_proba([processed])[0]
            best_index = probabilities.argmax()
            confidence = probabilities[best_index]
            intent = self.model.classes_[best_index]

            if confidence < 0.28:
                return "unknown_questions", confidence

            return intent, confidence

        except Exception:
            return "unknown_questions", 0

    def get_intent_response(self, tag):
        for intent in self.intents:
            if intent.get("tag") == tag:
                return random.choice(
                    intent.get("responses", ["I am still learning about that."])
                )
        return "I am still learning about that."

    def search_learned_answer(self, user_id, message):
        learned = get_learned_responses(user_id)

        if not learned:
            return None

        questions = [preprocess(item["question"]) for item in learned]
        user_question = preprocess(message)

        if not user_question:
            return None

        try:
            vectorizer = TfidfVectorizer()
            matrix = vectorizer.fit_transform(questions + [user_question])
            scores = cosine_similarity(matrix[-1], matrix[:-1])[0]

            best_index = scores.argmax()

            if scores[best_index] >= 0.50:
                return self.random_answer(learned[best_index]["answer"])

        except Exception:
            return None

        return None

    def parse_number(self, text):
        cleaned = (
            text.lower()
            .replace(",", "")
            .replace("/=", "")
            .replace("rs.", "")
            .replace("rs", "")
            .replace("lkr", "")
            .replace("isn", "is ")
            .replace("is", " ")
        )

        match = re.search(r"(\d+(?:\.\d+)?)", cleaned)
        return float(match.group(1)) if match else None

    def normalize_category(self, category):
        category = category.lower().strip()
        category = category.replace("my", "").strip()
        category = re.sub(r"[^a-zA-Z ]", "", category).strip()

        if not category:
            return "general"

        if any(x in category for x in ["food", "meal", "pizza", "restaurant", "dining"]):
            return "food"

        if any(x in category for x in ["transport", "bus", "taxi", "fuel", "travel"]):
            return "transport"

        if any(x in category for x in ["bill", "electricity", "water", "internet", "phone"]):
            return "bill"

        if any(x in category for x in ["rent", "room", "house rent"]):
            return "rent"

        if any(x in category for x in ["shopping", "clothes", "dress"]):
            return "shopping"

        if any(x in category for x in ["medical", "hospital", "medicine", "health"]):
            return "medical"

        if any(x in category for x in ["education", "school", "course", "book", "exam"]):
            return "education"

        if any(x in category for x in ["entertainment", "movie", "game"]):
            return "entertainment"

        return category

    def parse_expense(self, message):
        lower = message.lower().replace(",", "").replace("/=", "")

        patterns = [
            r"(?:i spent|spent|paid|add expense|record expense|expense)\s*(\d+(?:\.\d+)?)\s*(?:on|for)?\s*([a-zA-Z ]+)",
            r"(?:my\s*)?([a-zA-Z ]+)\s*(?:expense|expanse|cost)\s*(?:is)?\s*(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*(?:on|for)\s*([a-zA-Z ]+)"
        ]

        for pattern in patterns:
            match = re.search(pattern, lower)

            if match:
                if match.group(1).replace(".", "").isdigit():
                    amount = float(match.group(1))
                    category = match.group(2).strip()
                else:
                    category = match.group(1).strip()
                    amount = float(match.group(2))

                return amount, self.normalize_category(category)

        return None, None

    def monthly_summary_text(self, user_id):
        rows = get_monthly_expenses(user_id)
        budget = get_budget(user_id)
        profile = get_user_profile(user_id)

        salary = float(profile.get("salary") or 0)
        savings = float(profile.get("savings") or 0)

        if not rows:
            return "📈 Monthly Financial Overview\n\nNo expenses recorded for this month yet."

        total = sum(row["total"] for row in rows)
        top = rows[0]

        lines = []
        lines.append("📊 Monthly Financial Overview")
        lines.append("")
        lines.append(f"💸 Total Expenses: Rs.{total:,.0f}")

        if salary:
            lines.append(f"💰 Monthly Income: Rs.{salary:,.0f}")
            lines.append(f"✅ Remaining Balance: Rs.{salary - total:,.0f}")

        if savings:
            lines.append(f"🏦 Monthly Savings: Rs.{savings:,.0f}")

        lines.append("")
        lines.append("📂 Spending Breakdown")

        for row in rows:
            lines.append(f"• {row['category'].title()}: Rs.{row['total']:,.0f}")

        lines.append("")
        lines.append(f"🏆 Highest Spending Category: {top['category'].title()}")

        if budget:
            remaining = budget - total
            lines.append(f"📌 Monthly Budget: Rs.{budget:,.0f}")

            if remaining >= 0:
                lines.append(f"✅ Remaining Budget: Rs.{remaining:,.0f}")
            else:
                lines.append(f"⚠️ Budget Exceeded By: Rs.{abs(remaining):,.0f}")

        if salary:
            ratio = (total / salary) * 100
            lines.append("")
            lines.append(f"📊 Expense Ratio: {ratio:.1f}% of income")

        return "\n".join(lines)

    def recommendation(self, user_id):
        rows = get_monthly_expenses(user_id)
        profile = get_user_profile(user_id)

        salary = float(profile.get("salary") or 0)
        savings = float(profile.get("savings") or 0)

        if not rows:
            return (
                "I can give more accurate recommendations once your expenses are recorded.\n"
                "For now, focus on tracking your major monthly costs such as food, rent, transport, bills, and shopping."
            )

        total = sum(r["total"] for r in rows)
        top = rows[0]
        share = (top["total"] / total) * 100 if total else 0
        tip = get_random_tip()

        advice = []

        if salary:
            expense_ratio = (total / salary) * 100
            saving_ratio = (savings / salary) * 100 if savings else 0

            if expense_ratio > 80:
                advice.append("Your expenses are high compared to your income.")
            elif expense_ratio > 60:
                advice.append("Your spending level is moderate. Reducing flexible expenses can improve savings.")
            else:
                advice.append("Your expense level looks financially manageable.")

            if savings:
                if saving_ratio < 10:
                    advice.append("Your savings rate is low. Aim for at least 10% to 20% of income.")
                else:
                    advice.append("Your savings habit is positive. Keep building it consistently.")
            else:
                advice.append("Share your monthly savings so I can calculate your savings ratio.")

        if share > 40:
            advice.append(f"{top['category'].title()} takes {share:.1f}% of your spending. This category may need review.")
        else:
            advice.append("Your spending distribution looks reasonably balanced.")

        advice.append(f"Tip: {tip}")
        return " ".join(advice)

    def finance_term_answer(self, message):
        lower = message.lower()

        for term, explanation in FINANCE_TERMS.items():
            if term in lower:
                return explanation

        return None

    # ===============================
    # FINANCE CALCULATION FEATURES
    # ===============================
    def safe_calculate(self, expression):
        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
        }

        def calculate(node):
            if isinstance(node, ast.Num):
                return node.n

            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value

            if isinstance(node, ast.BinOp):
                return operators[type(node.op)](
                    calculate(node.left),
                    calculate(node.right)
                )

            if isinstance(node, ast.UnaryOp):
                return operators[type(node.op)](calculate(node.operand))

            raise ValueError("Invalid calculation")

        tree = ast.parse(expression, mode="eval")
        return calculate(tree.body)

    def quick_calculator(self, message):
        text = message.lower().replace(",", "").replace("/=", "")

        if not any(op in text for op in ["+", "-", "*", "x", "/", "÷", "plus", "minus", "multiply", "divide"]):
            return None

        expression = text
        expression = expression.replace("plus", "+")
        expression = expression.replace("minus", "-")
        expression = expression.replace("multiply", "*")
        expression = expression.replace("multiplied by", "*")
        expression = expression.replace("into", "*")
        expression = expression.replace("x", "*")
        expression = expression.replace("divide", "/")
        expression = expression.replace("divided by", "/")
        expression = expression.replace("÷", "/")

        expression = re.sub(r"[^0-9+\-*/(). ]", " ", expression)
        expression = re.sub(r"\s+", " ", expression).strip()

        if not expression:
            return None

        try:
            result = self.safe_calculate(expression)

            return (
                "🧮 Quick Calculation\n\n"
                f"Calculation: {expression}\n"
                f"Answer: Rs.{result:,.2f}"
            )

        except Exception:
            return None

    def loan_calculator(self, message):
        lower = message.lower().replace(",", "").replace("/=", "")

        if not any(word in lower for word in ["loan", "interest", "emi"]):
            return None

        amount_match = re.search(r"(\d+(?:\.\d+)?)", lower)
        interest_match = re.search(r"(\d+(?:\.\d+)?)\s*%", lower)
        month_match = re.search(r"(\d+)\s*(?:month|months)", lower)

        if not amount_match or not interest_match or not month_match:
            return (
                "Please provide loan amount, interest percentage, and duration.\n\n"
                "Example: 1000000 loan at 5% interest for 10 months"
            )

        principal = float(amount_match.group(1))
        interest_rate = float(interest_match.group(1))
        months = int(month_match.group(1))

        total_interest = principal * (interest_rate / 100)
        total_payment = principal + total_interest
        monthly_payment = total_payment / months

        return (
            "💰 Loan Calculation\n\n"
            f"Loan Amount: Rs.{principal:,.0f}\n"
            f"Interest Rate: {interest_rate}%\n"
            f"Duration: {months} months\n\n"
            f"📌 Monthly Payment: Rs.{monthly_payment:,.0f}\n"
            f"📈 Total Interest: Rs.{total_interest:,.0f}\n"
            f"💵 Total Payment: Rs.{total_payment:,.0f}"
        )

    def discount_calculator(self, message):
        lower = message.lower().replace(",", "").replace("/=", "")

        if "discount" not in lower:
            return None

        price_match = re.search(r"(\d+(?:\.\d+)?)", lower)
        discount_match = re.search(r"(\d+(?:\.\d+)?)\s*%", lower)

        if not price_match or not discount_match:
            return "Example: 5000 with 20% discount"

        price = float(price_match.group(1))
        discount_rate = float(discount_match.group(1))

        discount_amount = price * discount_rate / 100
        final_price = price - discount_amount

        return (
            "🏷️ Discount Calculation\n\n"
            f"Original Price: Rs.{price:,.0f}\n"
            f"Discount: {discount_rate}%\n"
            f"Discount Amount: Rs.{discount_amount:,.0f}\n\n"
            f"Final Price: Rs.{final_price:,.0f}"
        )

    def split_bill_calculator(self, message):
        lower = message.lower().replace(",", "").replace("/=", "")

        if "split" not in lower:
            return None

        numbers = re.findall(r"\d+(?:\.\d+)?", lower)

        if len(numbers) < 2:
            return "Example: Total bill 4500 split for 5 people"

        total_bill = float(numbers[0])
        people = int(float(numbers[1]))

        if people <= 0:
            return "Number of people must be greater than 0."

        each_person = total_bill / people

        return (
            "👥 Split Bill Calculation\n\n"
            f"Total Bill: Rs.{total_bill:,.0f}\n"
            f"People: {people}\n\n"
            f"Each Person Should Pay: Rs.{each_person:,.0f}"
        )

    def savings_goal_calculator(self, message):
        lower = message.lower().replace(",", "").replace("/=", "")

        if not any(word in lower for word in ["save", "saving goal", "savings goal"]):
            return None

        numbers = re.findall(r"\d+(?:\.\d+)?", lower)
        month_match = re.search(r"(\d+)\s*(?:month|months)", lower)

        if len(numbers) < 1 or not month_match:
            return "Example: I want to save 120000 in 6 months"

        goal_amount = float(numbers[0])
        months = int(month_match.group(1))

        monthly_saving = goal_amount / months

        return (
            "🎯 Savings Goal Calculation\n\n"
            f"Savings Goal: Rs.{goal_amount:,.0f}\n"
            f"Duration: {months} months\n\n"
            f"You should save Rs.{monthly_saving:,.0f} per month."
        )

    def emergency_fund_calculator(self, message):
        lower = message.lower().replace(",", "").replace("/=", "")

        if "emergency fund" not in lower:
            return None

        amount = self.parse_number(lower)

        if not amount:
            return "Example: emergency fund for monthly expense 40000"

        three_months = amount * 3
        six_months = amount * 6

        return (
            "🛡️ Emergency Fund Calculation\n\n"
            f"Monthly Expense: Rs.{amount:,.0f}\n\n"
            f"3 Months Emergency Fund: Rs.{three_months:,.0f}\n"
            f"6 Months Emergency Fund: Rs.{six_months:,.0f}"
        )

    def daily_spending_limit(self, user_id, message):
        lower = message.lower()

        if not any(x in lower for x in ["daily limit", "daily spending", "per day spend"]):
            return None

        budget = get_budget(user_id)
        rows = get_monthly_expenses(user_id)

        if not budget:
            return "Please set your monthly budget first. Example: set budget 50000"

        total_spent = sum(row["total"] for row in rows)
        remaining = budget - total_spent

        days_match = re.search(r"(\d+)\s*(?:day|days)", lower)
        days = int(days_match.group(1)) if days_match else 30

        if days <= 0:
            return "Remaining days must be greater than 0."

        daily_limit = remaining / days

        return (
            "📅 Daily Spending Limit\n\n"
            f"Monthly Budget: Rs.{budget:,.0f}\n"
            f"Total Spent: Rs.{total_spent:,.0f}\n"
            f"Remaining Budget: Rs.{remaining:,.0f}\n"
            f"Remaining Days: {days}\n\n"
            f"You can spend about Rs.{daily_limit:,.0f} per day."
        )

    def salary_handler(self, user_id, message):
        if any(word in message for word in ["salary", "income", "monthly income", "my pay"]):
            amount = self.parse_number(message)

            if amount:
                save_user_profile(user_id, salary=amount)

                return (
                    f"✨ Great! Your monthly income of Rs.{amount:,.0f} has been recorded successfully.\n\n"
                    "I can now help you build a smarter financial plan based on your income.\n\n"
                    "Next, share your regular monthly expenses such as food, rent, transportation, bills, shopping, or entertainment."
                )

        return None

    def savings_handler(self, user_id, message):
        if any(word in message for word in ["saving", "savings", "save monthly", "saved"]):
            amount = self.parse_number(message)

            if amount:
                save_user_profile(user_id, savings=amount)

                profile = get_user_profile(user_id)
                salary = float(profile.get("salary") or 0)

                if salary:
                    saving_ratio = (amount / salary) * 100

                    return (
                        f"💰 Excellent! Your monthly savings of Rs.{amount:,.0f} has been recorded.\n\n"
                        f"Your current savings rate is {saving_ratio:.1f}% of your monthly income.\n\n"
                        "This helps me give better budget, savings, and purchase advice."
                    )

                return (
                    f"💰 Excellent! Your monthly savings of Rs.{amount:,.0f} has been recorded.\n\n"
                    "Consistent savings are an important step toward financial stability."
                )

        return None

    def budget_recommendation(self, user_id):
        profile = get_user_profile(user_id)
        salary = float(profile.get("salary") or 0)

        if salary <= 0:
            return (
                "To prepare a personalized budget recommendation, please share your monthly income first.\n\n"
                "Example: my salary is 120000"
            )

        needs = salary * 0.50
        wants = salary * 0.30
        savings = salary * 0.20

        return (
            "📊 Personalized Monthly Budget Recommendation\n\n"
            f"Monthly Income: Rs.{salary:,.0f}\n\n"
            f"Essential Needs 50%: Rs.{needs:,.0f}\n"
            f"Lifestyle Wants 30%: Rs.{wants:,.0f}\n"
            f"Savings & Future Goals 20%: Rs.{savings:,.0f}\n\n"
            "This is a recommended starting point. It can be adjusted based on rent, family responsibilities, debt payments, and long-term goals."
        )

    def financial_health_score(self, user_id):
        rows = get_monthly_expenses(user_id)
        profile = get_user_profile(user_id)

        salary = float(profile.get("salary") or 0)
        savings = float(profile.get("savings") or 0)
        total_expense = sum(row["total"] for row in rows)

        if salary <= 0:
            return (
                "I can calculate your financial health score after you share your monthly income.\n\n"
                "Example: my salary is 120000"
            )

        score = 100
        expense_ratio = (total_expense / salary) * 100 if total_expense else 0
        saving_ratio = (savings / salary) * 100 if savings else 0

        if expense_ratio > 90:
            score -= 35
        elif expense_ratio > 75:
            score -= 25
        elif expense_ratio > 60:
            score -= 15

        if saving_ratio < 5:
            score -= 25
        elif saving_ratio < 10:
            score -= 15
        elif saving_ratio < 20:
            score -= 5

        if not rows:
            score -= 10

        score = max(0, min(100, score))

        if score >= 80:
            status = "Excellent"
        elif score >= 60:
            status = "Good"
        elif score >= 40:
            status = "Needs Improvement"
        else:
            status = "Risky"

        return (
            "📊 Your Financial Health Analysis\n\n"
            f"Score: {score}/100\n"
            f"Status: {status}\n"
            f"Expense Ratio: {expense_ratio:.1f}%\n"
            f"Savings Ratio: {saving_ratio:.1f}%\n\n"
            "To improve your score, reduce unnecessary expenses, increase savings gradually, and follow a realistic monthly budget."
        )

    def goal_advice(self, user_id, message):
        goals = ["iphone", "phone", "bike", "bicycle", "car", "house", "laptop", "gold", "land"]

        if not any(word in message for word in ["need", "buy", "want", "purchase", "afford"]):
            return None

        selected_goal = None

        for goal in goals:
            if goal in message:
                selected_goal = goal
                break

        if not selected_goal:
            return None

        profile = get_user_profile(user_id)
        salary = float(profile.get("salary") or 0)
        savings = float(profile.get("savings") or 0)

        rows = get_monthly_expenses(user_id)
        total_expense = sum(row["total"] for row in rows)

        if salary == 0:
            return (
                f"I can help you plan for a {selected_goal} professionally.\n\n"
                "To check affordability, please share your monthly income first."
            )

        if savings == 0:
            return (
                f"Your monthly income is Rs.{salary:,.0f}.\n\n"
                f"To give accurate advice about buying a {selected_goal}, please share your monthly savings as well."
            )

        saving_rate = (savings / salary) * 100 if salary else 0

        base = (
            f"Based on your income of Rs.{salary:,.0f}, monthly savings of Rs.{savings:,.0f}, "
            f"and recorded expenses of Rs.{total_expense:,.0f}, "
        )

        if selected_goal in ["iphone", "phone", "laptop"]:
            if saving_rate < 10:
                return (
                    base +
                    f"buying a {selected_goal} right now may put pressure on your budget.\n\n"
                    f"Your savings rate is {saving_rate:.1f}%. I recommend increasing savings first before making this purchase."
                )

            return (
                base +
                f"buying a {selected_goal} looks manageable if your emergency fund is safe and you avoid unnecessary debt."
            )

        if selected_goal == "bicycle":
            return (
                base +
                "a bicycle is a practical low-cost goal. It can also help reduce transport expenses."
            )

        if selected_goal == "bike":
            return (
                base +
                "a bike may be manageable if you plan for fuel, insurance, service, and EMI carefully. Keep EMI below 15% to 20% of income."
            )

        if selected_goal == "car":
            return (
                base +
                "a car is a major financial decision. Save for a down payment first and keep total vehicle costs under control."
            )

        if selected_goal == "house":
            return (
                base +
                "a house is a long-term goal. Build a separate house fund, save for down payment, and avoid unaffordable loans."
            )

        if selected_goal == "land":
            return (
                base +
                "land can be a long-term investment. Check documents, location, legal clearance, and affordability before buying."
            )

        if selected_goal == "gold":
            return (
                base +
                "gold can be useful as a long-term asset, but it is better to buy gradually using savings instead of debt."
            )

        return (
            base +
            f"before buying a {selected_goal}, check whether it is a need or want and protect your emergency savings."
        )

    def reminder_handler(self, user_id, message):
        if "show reminders" in message or "my reminders" in message:
            rows = get_reminders(user_id)

            if not rows:
                return "You have no bill reminders yet."

            text = "🔔 Your Bill Reminders:\n\n"
            for row in rows:
                text += f"• {row['bill_name']} - every month on {row['due_day']}\n"

            return text

        if "remind" in message or "reminder" in message:
            day_match = re.search(r"(\d{1,2})(?:st|nd|rd|th)?", message)

            if not day_match:
                return "Please mention the due date. Example: Remind me to pay electricity bill on 25th"

            due_day = int(day_match.group(1))

            if "electricity" in message:
                bill_name = "Electricity Bill"
            elif "rent" in message or "house rent" in message:
                bill_name = "House Rent"
            elif "credit card" in message:
                bill_name = "Credit Card Bill"
            elif "emi" in message:
                bill_name = "EMI Payment"
            else:
                bill_name = "Bill Payment"

            add_reminder(user_id, bill_name, due_day)

            return f"🔔 Reminder saved: {bill_name} on day {due_day} of every month."

        return None

    def reply(self, user_id, message):
        message = message.lower().strip()

        if not message:
            return {"reply": "Please enter your financial question or expense details 😊", "intent": "empty"}

        if message.startswith("teach:"):
            content = message[6:].strip()

            if "=>" in content:
                question, answer = content.split("=>", 1)
                save_learned_response(user_id, question.strip(), answer.strip())

                return {
                    "reply": "Done. I have learned this response and it is now available for users.",
                    "intent": "self_learning"
                }

            return {
                "reply": "Please use this format: teach: your question => the correct answer",
                "intent": "self_learning_help"
            }

        salary_reply = self.salary_handler(user_id, message)
        if salary_reply:
            return {"reply": salary_reply, "intent": "salary_saved"}

        savings_reply = self.savings_handler(user_id, message)
        if savings_reply:
            return {"reply": savings_reply, "intent": "savings_saved"}

        reminder_reply = self.reminder_handler(user_id, message)
        if reminder_reply:
            return {"reply": reminder_reply, "intent": "bill_reminder"}

        loan_reply = self.loan_calculator(message)
        if loan_reply:
            return {"reply": loan_reply, "intent": "loan_calculation"}

        quick_calc_reply = self.quick_calculator(message)
        if quick_calc_reply:
            return {"reply": quick_calc_reply, "intent": "quick_calculation"}

        discount_reply = self.discount_calculator(message)
        if discount_reply:
            return {"reply": discount_reply, "intent": "discount_calculation"}

        split_reply = self.split_bill_calculator(message)
        if split_reply:
            return {"reply": split_reply, "intent": "split_bill"}

        savings_goal_reply = self.savings_goal_calculator(message)
        if savings_goal_reply:
            return {"reply": savings_goal_reply, "intent": "savings_goal"}

        emergency_reply = self.emergency_fund_calculator(message)
        if emergency_reply:
            return {"reply": emergency_reply, "intent": "emergency_fund"}

        daily_limit_reply = self.daily_spending_limit(user_id, message)
        if daily_limit_reply:
            return {"reply": daily_limit_reply, "intent": "daily_spending_limit"}

        amount, category = self.parse_expense(message)
        if amount and category:
            add_expense(user_id, amount, category, message)

            profile = get_user_profile(user_id)
            budget = get_budget(user_id)
            rows = get_monthly_expenses(user_id)
            salary = float(profile.get("salary") or 0)
            total = sum(row["total"] for row in rows)

            warning = ""

            if budget:
                remaining = budget - total

                if remaining < 0:
                    warning = (
                        f"\n\n🚨 Budget Alert!\n"
                        f"You exceeded your monthly budget by Rs.{abs(remaining):,.0f}.\n"
                        "Please reduce unnecessary spending."
                    )

                elif remaining <= budget * 0.2:
                    warning += (
                        f"\n\n⚠️ Budget Warning!\n"
                        f"Only Rs.{remaining:,.0f} remaining in your monthly budget.\n"
                        "Be careful with extra spending."
                    )

            if salary:
                remaining_income = salary - total

                if remaining_income < 0:
                    warning += (
                        f"\n\n🚨 Spending Alert!\n"
                        f"You exceeded your monthly income by Rs.{abs(remaining_income):,.0f}."
                    )
                elif remaining_income <= salary * 0.1:
                    warning += (
                        f"\n\n⚠️ Spending Warning!\n"
                        f"Only Rs.{remaining_income:,.0f} remaining from your monthly income."
                    )

            return {
                "reply": (
                    f"✅ Expense recorded successfully.\n\n"
                    f"Category: {category.title()}\n"
                    f"Amount: Rs.{amount:,.0f}"
                    f"{warning}\n\n"
                    f"{self.recommendation(user_id)}"
                ),
                "intent": "add_expense"
            }

        budget_match = re.search(r"set budget\s*(\d+)", message)

        if budget_match:
            amount = float(budget_match.group(1))
            set_budget(user_id, amount)

            return {
                "reply": (
                    f"✅ Monthly budget set successfully.\n\n"
                    f"Budget Amount: Rs.{amount:,.0f}"
                ),
                "intent": "set_budget"
            }

        if any(x in message for x in ["budget warning", "warning", "budget alert"]):
            budget = get_budget(user_id)
            rows = get_monthly_expenses(user_id)
            total = sum(row["total"] for row in rows)
            remaining = budget - total

            if remaining <= budget * 0.2:
                return {
                    "reply": (
                        "⚠️ Budget Warning\n\n"
                        "You are close to your monthly budget limit.\n\n"
                        f"Remaining Budget: Rs.{remaining:,.0f}\n\n"
                        "Please reduce unnecessary spending."
                    ),
                    "intent": "budget_warning"
                }

            return {
                "reply": (
                    "✅ Budget Safe\n\n"
                    f"You still have Rs.{remaining:,.0f} remaining.\n\n"
                    "Your spending is under control."
                ),
                "intent": "budget_warning"
            }

        if any(x in message for x in ["monthly summary", "show summary", "my summary", "expense summary", "show my expenses"]):
            return {"reply": self.monthly_summary_text(user_id), "intent": "monthly_summary"}

        if any(x in message for x in ["make budget", "budget plan", "budget recommendation", "divide salary", "salary split"]):
            return {"reply": self.budget_recommendation(user_id), "intent": "budget_recommendation"}

        if any(x in message for x in ["financial score", "health score", "financial health"]):
            return {"reply": self.financial_health_score(user_id), "intent": "financial_health_score"}

        goal_reply = self.goal_advice(user_id, message)
        if goal_reply:
            return {"reply": goal_reply, "intent": "goal_advice"}

        if any(x in message for x in ["save money", "saving tips", "how can i save", "finance advice", "money advice"]):
            return {
                "reply": (
                    "💡 Saving Tips\n\n"
                    "• Track your daily expenses.\n"
                    "• Avoid unnecessary spending.\n"
                    "• Set a monthly budget.\n"
                    "• Save at least 10% to 20% of your income.\n"
                    "• Reduce food, shopping, and transport overspending.\n"
                    "• Keep emergency savings for unexpected needs."
                ),
                "intent": "saving_tips"
            }

        if any(word in message for word in ["recommend", "suggest", "smart advice", "overspending", "analyze"]):
            return {"reply": self.recommendation(user_id), "intent": "recommendation"}

        term_answer = self.finance_term_answer(message)
        if term_answer:
            return {"reply": term_answer, "intent": "finance_terms"}

        intent, confidence = self.predict_intent(message)

        if intent == "add_expense":
            amount, category = self.parse_expense(message)

            if amount:
                add_expense(user_id, amount, category, message)

                return {
                    "reply": (
                        f"✅ Expense recorded successfully.\n\n"
                        f"Category: {category.title()}\n"
                        f"Amount: Rs.{amount:,.0f}\n\n"
                        f"{self.recommendation(user_id)}"
                    ),
                    "intent": intent
                }

            return {
                "reply": "Please include the amount and category so I can record the expense correctly.",
                "intent": intent
            }

        if intent == "budget_help":
            amount = self.parse_number(message)

            if amount:
                set_budget(user_id, amount)

                return {
                    "reply": f"📌 Your monthly budget has been set to Rs.{amount:,.0f}.",
                    "intent": intent
                }

            return {"reply": self.budget_recommendation(user_id), "intent": intent}

        if intent == "monthly_summary":
            return {"reply": self.monthly_summary_text(user_id), "intent": "monthly_summary"}

        if intent == "saving_tips":
            return {"reply": self.recommendation(user_id), "intent": "saving_tips"}

        if intent == "finance_terms":
            answer = self.finance_term_answer(message)
            if answer:
                return {"reply": answer, "intent": "finance_terms"}

        learned_answer = self.search_learned_answer(user_id, message)
        if learned_answer:
            return {"reply": learned_answer, "intent": "learned_response"}

        if intent == "unknown_questions":
            return {
                "reply": (
                    "I’m still learning about that topic 😊\n\n"
                    "You can ask me about budgeting, expenses, savings, investments, loans, buying decisions, or financial planning."
                ),
                "intent": intent
            }

        return {
            "reply": self.get_intent_response(intent),
            "intent": intent,
            "confidence": confidence
        }