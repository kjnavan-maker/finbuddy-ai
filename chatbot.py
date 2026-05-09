"""NLP, inference engine, knowledge-base search, and self-learning logic for FinBuddy AI."""

import json
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
    get_user_profile
)

from model_training import train_model


MODEL_FILE = "intent_model.joblib"
DATA_FILE = os.path.join("dataset", "intents.json")

lemmatizer = WordNetLemmatizer()


FINANCE_TERMS = {
    "budget": "A budget is a financial plan for managing income, expenses, savings, and goals.",
    "inflation": "Inflation is the increase in prices over time, which reduces purchasing power.",
    "interest": "Interest is the cost of borrowing money or the reward for saving money.",
    "compound interest": "Compound interest means earning interest on both original money and previous interest.",
    "emergency fund": "An emergency fund is money saved for unexpected situations.",
    "debt": "Debt is borrowed money that must be repaid.",
    "savings": "Savings are money kept aside for future needs instead of being spent immediately.",
    "investment": "Investment means using money to buy assets that may grow in value over time.",
    "expense": "An expense is money spent on needs, wants, bills, or services.",
    "emi": "EMI means Equated Monthly Installment, a fixed monthly loan repayment.",
    "loan": "A loan is borrowed money that must be repaid, usually with interest.",
    "asset": "An asset is something valuable that you own.",
    "liability": "A liability is money you owe to others.",
    "net worth": "Net worth is total assets minus total liabilities.",
    "nlp": "NLP means Natural Language Processing, which helps computers understand human language.",
    "knowledge base": "A knowledge base stores facts, learned answers, finance tips, expenses, and budgets.",
    "inference engine": "An inference engine uses rules and user data to choose the best reply or advice."
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
            text.replace(",", "")
            .replace("/=", "")
            .replace("rs.", "")
            .replace("rs", "")
            .replace("lkr", "")
        )
        match = re.search(r"(\d+(?:\.\d+)?)", cleaned)
        return float(match.group(1)) if match else None

    def normalize_category(self, category):
        category = category.lower().strip()
        category = category.replace("my", "").strip()
        category = re.sub(r"[^a-zA-Z ]", "", category).strip()

        if not category:
            return "general"

        if "food" in category or "meal" in category or "pizza" in category or "restaurant" in category:
            return "food"

        if "transport" in category or "bus" in category or "taxi" in category or "fuel" in category:
            return "transport"

        if "bill" in category or "electricity" in category or "water" in category or "internet" in category or "phone" in category:
            return "bill"

        if "rent" in category or "room" in category or "house rent" in category:
            return "rent"

        if "shopping" in category or "clothes" in category or "dress" in category:
            return "shopping"

        if "medical" in category or "hospital" in category or "medicine" in category:
            return "medical"

        if "education" in category or "school" in category or "course" in category or "book" in category:
            return "education"

        if "entertainment" in category or "movie" in category or "game" in category:
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

                category = self.normalize_category(category)
                return amount, category

        return None, None

    def monthly_summary_text(self, user_id):
        rows = get_monthly_expenses(user_id)
        budget = get_budget(user_id)
        profile = get_user_profile(user_id)

        salary = float(profile.get("salary") or 0)
        savings = float(profile.get("savings") or 0)

        if not rows:
            return "You have not recorded expenses this month yet. Try: I spent 500 on food."

        total = sum(row["total"] for row in rows)
        top = rows[0]

        lines = []
        lines.append("📊 Monthly Finance Summary")
        lines.append("")
        lines.append(f"Total Expenses: Rs.{total:,.0f}")

        if salary:
            remaining_after_expenses = salary - total
            lines.append(f"Salary: Rs.{salary:,.0f}")
            lines.append(f"Balance after expenses: Rs.{remaining_after_expenses:,.0f}")

        if savings:
            lines.append(f"Saved Amount: Rs.{savings:,.0f}")

        lines.append("")
        lines.append("Category Breakdown:")

        for row in rows:
            lines.append(f"- {row['category'].title()}: Rs.{row['total']:,.0f}")

        lines.append("")
        lines.append(f"Highest Spending Category: {top['category'].title()}")

        if budget:
            remaining = budget - total

            if remaining >= 0:
                lines.append(f"Budget: Rs.{budget:,.0f}")
                lines.append(f"Remaining Budget: Rs.{remaining:,.0f}")
            else:
                lines.append(f"Budget: Rs.{budget:,.0f}")
                lines.append(f"⚠️ You exceeded your budget by Rs.{abs(remaining):,.0f}")

        if salary:
            expense_ratio = (total / salary) * 100
            lines.append("")
            lines.append(f"Expense Ratio: {expense_ratio:.1f}% of salary")

            if expense_ratio > 80:
                lines.append("⚠️ Your expenses are very high. Try reducing non-essential spending.")
            elif expense_ratio > 60:
                lines.append("Your expenses are moderate-high. Try improving your savings rate.")
            else:
                lines.append("Good job. Your expenses look controlled compared to salary.")

        return "\n".join(lines)

    def recommendation(self, user_id):
        rows = get_monthly_expenses(user_id)
        profile = get_user_profile(user_id)

        salary = float(profile.get("salary") or 0)
        savings = float(profile.get("savings") or 0)

        if not rows:
            return "Start by tracking your expenses. Then I can give better personalized recommendations."

        total = sum(r["total"] for r in rows)
        top = rows[0]
        share = (top["total"] / total) * 100 if total else 0
        tip = get_random_tip()

        advice = []

        if salary:
            expense_ratio = (total / salary) * 100
            saving_ratio = (savings / salary) * 100 if savings else 0

            if expense_ratio > 80:
                advice.append("⚠️ Your expenses are very high compared to your salary.")
            elif expense_ratio > 60:
                advice.append("Your expenses are a little high. Try reducing wants.")
            else:
                advice.append("Your expense level looks manageable.")

            if savings:
                if saving_ratio < 10:
                    advice.append("Your savings rate is low. Try saving at least 10% to 20% of salary.")
                else:
                    advice.append("Your savings habit is good. Keep improving it.")
            else:
                advice.append("Add your savings amount so I can calculate your savings rate.")

        if share > 40:
            advice.append(
                f"{top['category'].title()} is {share:.1f}% of your spending. Try reducing this category."
            )
        else:
            advice.append("Your spending is reasonably balanced.")

        advice.append(f"Extra tip: {tip}")

        return " ".join(advice)

    def finance_term_answer(self, message):
        lower = message.lower()

        for term, explanation in FINANCE_TERMS.items():
            if term in lower:
                return explanation

        return None

    def salary_handler(self, user_id, message):
        salary_keywords = ["salary", "income", "monthly income", "my pay"]

        if any(word in message for word in salary_keywords):
            amount = self.parse_number(message)

            if amount:
                save_user_profile(user_id, salary=amount)
                return (
                    f"✅ Your salary Rs.{amount:,.0f} has been saved.\n"
                    "Now add your expenses like: food expense 30000, rent 25000, transport expense 20000."
                )

        return None

    def savings_handler(self, user_id, message):
        savings_keywords = ["saving", "savings", "save monthly", "saved"]

        if any(word in message for word in savings_keywords):
            amount = self.parse_number(message)

            if amount:
                save_user_profile(user_id, savings=amount)

                profile = get_user_profile(user_id)
                salary = float(profile.get("salary") or 0)

                if salary:
                    saving_ratio = (amount / salary) * 100
                    return (
                        f"✅ Your monthly savings Rs.{amount:,.0f} has been saved.\n"
                        f"Your savings rate is {saving_ratio:.1f}% of your salary."
                    )

                return f"✅ Your monthly savings Rs.{amount:,.0f} has been saved."

        return None

    def budget_recommendation(self, user_id):
        profile = get_user_profile(user_id)
        salary = float(profile.get("salary") or 0)

        if salary <= 0:
            return "Please tell me your salary first. Example: my salary is 120000"

        needs = salary * 0.50
        wants = salary * 0.30
        savings = salary * 0.20

        return (
            "💰 Suggested Monthly Budget Plan\n\n"
            f"Salary: Rs.{salary:,.0f}\n"
            f"Needs 50%: Rs.{needs:,.0f}\n"
            f"Wants 30%: Rs.{wants:,.0f}\n"
            f"Savings 20%: Rs.{savings:,.0f}\n\n"
            "You can adjust this based on rent, food, bills, debt, and family responsibilities."
        )

    def financial_health_score(self, user_id):
        rows = get_monthly_expenses(user_id)
        profile = get_user_profile(user_id)

        salary = float(profile.get("salary") or 0)
        savings = float(profile.get("savings") or 0)
        total_expense = sum(row["total"] for row in rows)

        if salary <= 0:
            return "Please add your salary first so I can calculate your financial health score."

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
            f"📈 Financial Health Score: {score}/100\n"
            f"Status: {status}\n"
            f"Expense Ratio: {expense_ratio:.1f}%\n"
            f"Savings Ratio: {saving_ratio:.1f}%\n\n"
            "To improve your score: reduce unnecessary expenses, increase savings, and follow a monthly budget."
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
                f"Sure, I can help you plan for a {selected_goal}. "
                "Please tell me your salary first. Example: my salary is 120000"
            )

        if savings == 0:
            return (
                f"I know your salary is Rs.{salary:,.0f}. "
                "Please tell me your monthly savings too. Example: my savings 10000"
            )

        available_balance = salary - total_expense - savings
        saving_rate = (savings / salary) * 100 if salary else 0

        base = (
            f"Based on your salary Rs.{salary:,.0f}, savings Rs.{savings:,.0f}, "
            f"and recorded expenses Rs.{total_expense:,.0f}, "
        )

        if selected_goal in ["iphone", "phone", "laptop"]:
            if saving_rate < 10:
                return (
                    base +
                    f"buying a {selected_goal} should be delayed. Your savings rate is only {saving_rate:.1f}%. "
                    "Try increasing savings to at least 10% to 20% before buying."
                )

            return (
                base +
                f"you can plan to buy a {selected_goal}, but avoid loans. "
                "Buy only if your emergency fund and monthly budget are safe."
            )

        if selected_goal == "bicycle":
            return (
                base +
                "a bicycle is a smart low-cost goal. It can reduce transport expenses. "
                "Try buying it using savings without a loan."
            )

        if selected_goal == "bike":
            return (
                base +
                "a bike is possible if EMI, fuel, insurance, and service costs fit your budget. "
                "Keep EMI below 15% to 20% of your salary."
            )

        if selected_goal == "car":
            return (
                base +
                "a car is a big financial decision. Save for a down payment first and keep EMI below 15% to 20% of salary."
            )

        if selected_goal == "house":
            return (
                base +
                "a house is a long-term goal. Build a house fund, save for down payment, and avoid unaffordable loans."
            )

        if selected_goal == "land":
            return (
                base +
                "land can be a long-term investment. Check legal documents, location, and affordability before buying."
            )

        if selected_goal == "gold":
            return (
                base +
                "gold can be useful as a long-term asset, but buy using savings, not debt."
            )

        return (
            base +
            f"before buying a {selected_goal}, check whether it is a need or want and protect emergency savings."
        )

    def reply(self, user_id, message):
        message = message.lower().strip()

        if not message:
            return {"reply": "Please type a message so I can help.", "intent": "empty"}

        # 1. Teach command
        if message.startswith("teach:"):
            content = message[6:].strip()

            if "=>" in content:
                question, answer = content.split("=>", 1)
                save_learned_response(user_id, question.strip(), answer.strip())

                return {
                    "reply": "Thank you! I learned that answer globally. Now all users can use it. 🧠",
                    "intent": "self_learning"
                }

            return {
                "reply": "Use this format: teach: your question => the correct answer",
                "intent": "self_learning_help"
            }

        # 2. Salary save
        salary_reply = self.salary_handler(user_id, message)
        if salary_reply:
            return {"reply": salary_reply, "intent": "salary_saved"}

        # 3. Savings save
        savings_reply = self.savings_handler(user_id, message)
        if savings_reply:
            return {"reply": savings_reply, "intent": "savings_saved"}

        # 4. Expense add
        amount, category = self.parse_expense(message)
        if amount and category:
            add_expense(user_id, amount, category, message)
            return {
                "reply": (
                    f"✅ Expense added successfully!\n"
                    f"Amount: Rs.{amount:,.0f}\n"
                    f"Category: {category.title()}\n\n"
                    f"{self.recommendation(user_id)}"
                ),
                "intent": "add_expense"
            }

        # 5. Budget set/check
        if "set budget" in message or message.startswith("budget ") or "my budget" in message:
            amount = self.parse_number(message)

            if amount:
                set_budget(user_id, amount)
                return {
                    "reply": f"✅ Monthly budget set to Rs.{amount:,.0f}. I will compare your spending with this budget.",
                    "intent": "budget_set"
                }

            budget = get_budget(user_id)

            if budget:
                return {
                    "reply": f"Your current monthly budget is Rs.{budget:,.0f}.\n\n{self.monthly_summary_text(user_id)}",
                    "intent": "budget_help"
                }

            return {
                "reply": "Tell me your budget like this: set budget 50000.",
                "intent": "budget_help"
            }

        # 6. Monthly summary
        if any(x in message for x in ["monthly summary", "show summary", "my summary", "expense summary", "show my expenses"]):
            return {"reply": self.monthly_summary_text(user_id), "intent": "monthly_summary"}

        # 7. Budget recommendation
        if any(x in message for x in ["make budget", "budget plan", "budget recommendation", "divide salary", "salary split"]):
            return {"reply": self.budget_recommendation(user_id), "intent": "budget_recommendation"}

        # 8. Financial health score
        if any(x in message for x in ["financial score", "health score", "financial health"]):
            return {"reply": self.financial_health_score(user_id), "intent": "financial_health_score"}

        # 9. Goal/buying advice
        goal_reply = self.goal_advice(user_id, message)
        if goal_reply:
            return {"reply": goal_reply, "intent": "goal_advice"}

        # 10. Recommendation / analysis
        if any(word in message for word in ["recommend", "suggest", "smart advice", "overspending", "analyze"]):
            return {"reply": self.recommendation(user_id), "intent": "recommendation"}

        # 11. Finance term direct answer
        term_answer = self.finance_term_answer(message)
        if term_answer:
            return {"reply": term_answer, "intent": "finance_terms"}

        # 12. ML intent
        intent, confidence = self.predict_intent(message)

        if intent == "add_expense":
            amount, category = self.parse_expense(message)

            if amount:
                add_expense(user_id, amount, category, message)
                return {
                    "reply": f"Done! I recorded Rs.{amount:,.0f} under {category.title()}. {self.recommendation(user_id)}",
                    "intent": intent
                }

            return {
                "reply": "Please include the amount, for example: I spent 500 on food.",
                "intent": intent
            }

        if intent == "budget_help":
            amount = self.parse_number(message)

            if amount:
                set_budget(user_id, amount)
                return {
                    "reply": f"Monthly budget set to Rs.{amount:,.0f}. I will compare your spending with this budget.",
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

        # 13. Learned/admin knowledge near bottom
        learned_answer = self.search_learned_answer(user_id, message)
        if learned_answer:
            return {"reply": learned_answer, "intent": "learned_response"}

        # 14. Unknown fallback
        if intent == "unknown_questions":
            return {
                "reply": (
                    "I am not fully sure about that yet. "
                    "Please tell me your salary, expenses, savings, budget, or goal so I can help better. "
                    "Admin can also teach me using: teach: your question => the correct answer"
                ),
                "intent": intent
            }

        return {
            "reply": self.get_intent_response(intent),
            "intent": intent,
            "confidence": confidence
        }