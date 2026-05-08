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
    "budget": "A budget is a plan for how you will spend and save your money during a period.",
    "inflation": "Inflation is the increase in prices over time, which reduces purchasing power.",
    "interest": "Interest is the cost of borrowing money or the reward for saving money.",
    "compound interest": "Compound interest means earning interest on both original money and previous interest.",
    "emergency fund": "An emergency fund is money saved for unexpected situations.",
    "debt": "Debt is money borrowed that must be repaid.",
    "savings": "Savings are money kept aside instead of being spent immediately.",
    "nlp": "NLP means Natural Language Processing, which helps computers understand human language.",
    "knowledge base": "A knowledge base stores facts, learned answers, finance tips, expenses, and budgets."
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
                return random.choice(intent.get("responses", ["I am still learning about that."]))

        return "I am still learning about that."

    def search_learned_answer(self, user_id, message):
        learned = get_learned_responses(user_id)

        if not learned:
            return None

        questions = [preprocess(item["question"]) for item in learned]
        user_question = preprocess(message)

        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(questions + [user_question])
        scores = cosine_similarity(matrix[-1], matrix[:-1])[0]

        best_index = scores.argmax()

        if scores[best_index] >= 0.45:
            return self.random_answer(learned[best_index]["answer"])

        return None

    def parse_number(self, text):
        cleaned = text.replace(",", "").replace("/=", "")
        match = re.search(r"(\d+(?:\.\d+)?)", cleaned)
        return float(match.group(1)) if match else None

    def parse_expense(self, message):
        lower = message.lower().replace(",", "").replace("/=", "")

        patterns = [
            r"(?:i spent|spent|paid|add expense|record expense|expense)\s*(\d+(?:\.\d+)?)\s*(?:on|for)?\s*([a-zA-Z ]+)",
            r"(?:my\s*)?([a-zA-Z ]+)\s*(?:expense|expanse)\s*(\d+(?:\.\d+)?)",
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

                category = category.replace("my", "").strip()
                category = re.sub(r"[^a-zA-Z ]", "", category).strip()

                if not category:
                    category = "general"

                if "bill" in category:
                    category = "bill"
                elif "room rent" in category or "rent" in category:
                    category = "rent"
                elif "food" in category:
                    category = "food"
                elif "transport" in category:
                    category = "transport"
                elif "shopping" in category:
                    category = "shopping"

                return amount, category

        return None, None

    def monthly_summary_text(self, user_id):
        rows = get_monthly_expenses(user_id)
        budget = get_budget(user_id)

        if not rows:
            return "You have not recorded expenses this month yet. Try: I spent 500 on food."

        total = sum(row["total"] for row in rows)
        top = rows[0]

        lines = [f"This month you spent Rs.{total:,.0f}.", ""]

        for row in rows:
            lines.append(f"{row['category'].title()}: Rs.{row['total']:,.0f}")

        lines.append("")
        lines.append(f"Highest spending category: {top['category'].title()}.")

        if budget:
            remaining = budget - total
            if remaining >= 0:
                lines.append(f"Budget: Rs.{budget:,.0f}. Remaining: Rs.{remaining:,.0f}.")
            else:
                lines.append(f"Budget: Rs.{budget:,.0f}. You exceeded it by Rs.{abs(remaining):,.0f}.")

        return "\n".join(lines)

    def recommendation(self, user_id):
        rows = get_monthly_expenses(user_id)

        if not rows:
            return "Start by tracking your expenses. Then I can give better personalized recommendations."

        total = sum(r["total"] for r in rows)
        top = rows[0]
        share = (top["total"] / total) * 100 if total else 0
        tip = get_random_tip()

        if share > 40:
            return f"Smart recommendation: {top['category'].title()} is {share:.1f}% of your spending. Try reducing this category. Extra tip: {tip}"

        return f"Your spending is reasonably balanced. Extra tip: {tip}"

    def finance_term_answer(self, message):
        lower = message.lower()

        for term, explanation in FINANCE_TERMS.items():
            if term in lower:
                return explanation

        return None

    def salary_handler(self, user_id, message):
        if "salary" in message or "income" in message:
            amount = self.parse_number(message)

            if amount:
                save_user_profile(user_id, salary=amount)
                return f"Your salary Rs.{amount:,.0f} has been saved. I can now give better personalized financial advice."

        return None

    def savings_handler(self, user_id, message):
        if "saving" in message or "savings" in message:
            amount = self.parse_number(message)

            if amount:
                save_user_profile(user_id, savings=amount)
                return f"Your monthly savings Rs.{amount:,.0f} has been saved."

        return None

    def goal_advice(self, user_id, message):
        goals = ["iphone", "phone", "bike", "bicycle", "car", "house", "laptop", "gold", "land"]

        if not any(word in message for word in ["need", "buy", "want", "purchase"]):
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
            return f"Sure, I can help you plan for a {selected_goal}. Please tell me your salary first. Example: my salary is 120000"

        if savings == 0:
            return f"I know your salary is Rs.{salary:,.0f}. Please tell me your monthly savings too. Example: my savings 10000"

        balance = salary - total_expense - savings

        if selected_goal in ["iphone", "phone", "laptop"]:
            if savings < salary * 0.1:
                return f"Based on your salary Rs.{salary:,.0f}, your savings are low. Before buying a {selected_goal}, try increasing savings to at least 10% to 20% of your salary."
            return f"You can plan to buy a {selected_goal}, but avoid loans. Keep emergency savings safe and make sure your monthly budget is not affected."

        if selected_goal == "bicycle":
            return "A bicycle is a good low-cost goal. It can reduce transport expenses. Save monthly and buy without taking a loan if possible."

        if selected_goal == "bike":
            return f"For a bike, first save a good down payment. EMI should stay below 15% to 20% of your salary. Also consider fuel, insurance, and service costs."

        if selected_goal == "car":
            return f"A car is a big financial decision. With salary Rs.{salary:,.0f}, save for a down payment first and keep EMI below 15% to 20% of your salary."

        if selected_goal == "house":
            return "A house is a long-term goal. Start a separate house fund, save for a down payment, reduce debt, and avoid unaffordable loans."

        if selected_goal == "land":
            return "Land is a long-term investment. Check location, legal documents, price growth, and save for a down payment before buying."

        if selected_goal == "gold":
            return "Gold can be useful as a long-term asset, but buy using savings, not debt. Make sure emergency savings are ready first."

        return f"Before buying a {selected_goal}, check your salary, expenses, savings, emergency fund, and whether it is a need or want."

    def reply(self, user_id, message):
        message = message.lower().strip()

        if not message:
            return {"reply": "Please type a message so I can help.", "intent": "empty"}

        # Teach command first
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

        # Global learned knowledge
        learned_answer = self.search_learned_answer(user_id, message)
        if learned_answer:
            return {"reply": learned_answer, "intent": "learned_response"}

        # User-specific profile
        salary_reply = self.salary_handler(user_id, message)
        if salary_reply:
            return {"reply": salary_reply, "intent": "salary_saved"}

        savings_reply = self.savings_handler(user_id, message)
        if savings_reply:
            return {"reply": savings_reply, "intent": "savings_saved"}

        # Goal planning
        goal_reply = self.goal_advice(user_id, message)
        if goal_reply:
            return {"reply": goal_reply, "intent": "goal_advice"}

        lower = message.lower()

        if any(word in lower for word in ["recommend", "suggest", "smart advice", "overspending", "analyze"]):
            return {"reply": self.recommendation(user_id), "intent": "recommendation"}

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

        # Extra expense fallback
        amount, category = self.parse_expense(message)
        if amount and category:
            add_expense(user_id, amount, category, message)
            return {
                "reply": f"Expense added successfully! Rs.{amount:,.0f} added under {category.title()} category.",
                "intent": "add_expense"
            }

        if intent == "budget_help":
            amount = self.parse_number(message)

            if amount:
                set_budget(user_id, amount)
                return {
                    "reply": f"Monthly budget set to Rs.{amount:,.0f}. I will compare your spending with this budget.",
                    "intent": intent
                }

            budget = get_budget(user_id)

            if budget:
                return {
                    "reply": f"Your current monthly budget is Rs.{budget:,.0f}. {self.monthly_summary_text(user_id)}",
                    "intent": intent
                }

            return {"reply": "Tell me your budget like this: set budget 50000.", "intent": intent}

        if "set budget" in message or message.startswith("budget"):
            amount = self.parse_number(message)

            if amount:
                set_budget(user_id, amount)
                return {"reply": f"Monthly budget set to Rs.{amount:,.0f}.", "intent": "budget_help"}

        if intent == "monthly_summary" or "monthly summary" in message:
            return {"reply": self.monthly_summary_text(user_id), "intent": "monthly_summary"}

        if intent == "saving_tips" or "saving tip" in message or "save money" in message:
            return {"reply": self.recommendation(user_id), "intent": "saving_tips"}

        if intent == "finance_terms":
            answer = self.finance_term_answer(message)
            if answer:
                return {"reply": answer, "intent": "finance_terms"}

        term_answer = self.finance_term_answer(message)
        if term_answer:
            return {"reply": term_answer, "intent": "finance_terms"}

        if intent == "unknown_questions":
            return {
                "reply": "I am not fully sure about that yet. Admin can teach me using: teach: your question => the correct answer",
                "intent": intent
            }

        return {
            "reply": self.get_intent_response(intent),
            "intent": intent,
            "confidence": confidence
        }