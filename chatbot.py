"""Professional NLP, inference engine, finance logic, and self-learning for FinBuddy AI."""

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
            return (
                "📈 Monthly Financial Overview\n\n"
                "I don’t see any expenses recorded for this month yet.\n\n"
                "Once you share your monthly expenses, I can calculate your total spending, remaining budget, savings ratio, and financial health."
            )

        total = sum(row["total"] for row in rows)
        top = rows[0]

        lines = []
        lines.append("📈 Monthly Financial Overview")
        lines.append("")
        lines.append(f"Total Monthly Expenses: Rs.{total:,.0f}")

        if salary:
            balance_after_expenses = salary - total
            lines.append(f"Monthly Income: Rs.{salary:,.0f}")
            lines.append(f"Balance After Expenses: Rs.{balance_after_expenses:,.0f}")

        if savings:
            lines.append(f"Monthly Savings: Rs.{savings:,.0f}")

        lines.append("")
        lines.append("Spending Breakdown:")

        for row in rows:
            lines.append(f"• {row['category'].title()}: Rs.{row['total']:,.0f}")

        lines.append("")
        lines.append(f"Highest Spending Category: {top['category'].title()}")

        if budget:
            remaining = budget - total
            lines.append(f"Monthly Budget: Rs.{budget:,.0f}")

            if remaining >= 0:
                lines.append(f"Remaining Budget: Rs.{remaining:,.0f}")
            else:
                lines.append(f"⚠️ Budget Exceeded By: Rs.{abs(remaining):,.0f}")

        if salary:
            expense_ratio = (total / salary) * 100
            lines.append("")
            lines.append(f"Expense Ratio: {expense_ratio:.1f}% of income")

            if expense_ratio > 80:
                lines.append("Recommendation: Your expenses are very high. Try reducing non-essential spending.")
            elif expense_ratio > 60:
                lines.append("Recommendation: Your expenses are manageable, but savings can still be improved.")
            else:
                lines.append("Recommendation: Your spending looks well controlled compared to your income.")

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

    def reply(self, user_id, message):
        message = message.lower().strip()

        if not message:
            return {"reply": "Please enter your financial question or expense details 😊", "intent": "empty"}

        # 1. Teach command
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
                    f"✅ Expense recorded successfully.\n\n"
                    f"Category: {category.title()}\n"
                    f"Amount: Rs.{amount:,.0f}\n\n"
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
                    "reply": (
                        f"📌 Your monthly budget has been set to Rs.{amount:,.0f}.\n\n"
                        "I’ll compare your expenses with this budget and help you stay on track."
                    ),
                    "intent": "budget_set"
                }

            budget = get_budget(user_id)

            if budget:
                return {
                    "reply": (
                        f"Your current monthly budget is Rs.{budget:,.0f}.\n\n"
                        f"{self.monthly_summary_text(user_id)}"
                    ),
                    "intent": "budget_help"
                }

            return {
                "reply": "To set your monthly budget, type something like: set budget 50000.",
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

        # 13. Learned/admin knowledge near bottom
        learned_answer = self.search_learned_answer(user_id, message)
        if learned_answer:
            return {"reply": learned_answer, "intent": "learned_response"}

        # 14. Unknown fallback
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