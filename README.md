# CalcBot
Simple calculus tutor chat bot that explains core concepts, solves problems, and adjusts difficulty explanations based on user comprehension

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:27:32 2025

@author: zionalexander
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Offline AI Calculus Tutor
- ML classifier for topic prediction
- Sympy for math solving
- Adaptive explanations (simpler if needed)
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sympy import symbols, diff, integrate, limit, sin, cos, exp

# --- symbolic variable ---
x = symbols('x')

# --- Knowledge base with multiple explanation levels ---
concepts = {
    "derivative": {
        "explanations": [
            "A derivative measures how fast a function changes — the slope at a point.",
            "Think of it as 'how steep' the graph is at a certain x value.",
            "Imagine a car moving; the derivative is its speed at a particular moment."
        ],
        "example": "For f(x)=x**2, the derivative f'(x)=2*x."
    },
    "integral": {
        "explanations": [
            "An integral adds up tiny slices to find the total area under a curve.",
            "Think of slicing the area under the curve into thin strips and adding them.",
            "It’s like summing all the small steps to know how far you've walked."
        ],
        "example": "For f(x)=x, ∫x dx = (x**2)/2 + C."
    },
    "limit": {
        "explanations": [
            "A limit shows what value a function approaches as x gets close to something.",
            "Zoom in on the graph near a point; the limit is the value you approach.",
            "It’s like asking 'where will a ball land if I roll it very close to the edge?'"
        ],
        "example": "For f(x)=(x**2−1)/(x−1), as x→1, f(x)→2."
    }
}

# --- Training data for ML ---
training_questions = [
    "how do i find the derivative of x squared",
    "differentiate x**2",
    "what is a derivative",
    "find the integral of x squared",
    "integrate sin x",
    "what is an integral",
    "what is a limit",
    "find the limit as x approaches 2",
    "limit of (x**2 - 4)/(x - 2)"
]

training_labels = [
    "derivative","derivative","derivative",
    "integral","integral","integral",
    "limit","limit","limit"
]

# --- Train ML classifier ---
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(training_questions)
model = MultinomialNB()
model.fit(X_train, training_labels)

# --- Solve math function ---
def solve_math(topic, expression):
    try:
        if topic == "derivative":
            return diff(expression, x)
        elif topic == "integral":
            return integrate(expression, x)
        elif topic == "limit":
            return limit(expression, x, 2)  # default x->2
    except Exception:
        return None

# --- Adaptive explanation ---
def adaptive_explanation(topic):
    explanations = concepts[topic]["explanations"]
    for explanation in explanations:
        print(f"CalcBot: {explanation}")
        user_feedback = input("Do you understand? (yes/no): ").lower()
        if user_feedback in ["yes", "y"]:
            print(f"Example: {concepts[topic]['example']}")
            return
    print("CalcBot: Keep practicing or try a simpler example!")

# --- Main interactive loop ---
print("🤖 Welcome to Offline CalcBot!")
print("Ask about derivatives, integrals, or limits. Type 'quit' to exit.\n")

while True:
    question = input("You: ")
    if question.lower() in ["quit","exit"]:
        print("CalcBot: Goodbye! Keep practicing calculus.")
        break

    # Predict topic
    X_test = vectorizer.transform([question])
    predicted_topic = model.predict(X_test)[0]

    # Try extracting math expression from input
    expr = None
    words = question.replace("of","").replace("find","").split()
    for w in words:
        try:
            expr = eval(w, {"x": x, "sin": sin, "cos": cos, "exp": exp})
            break
        except Exception:
            continue

    # Respond with solution if expression exists
    if expr:
        solution = solve_math(predicted_topic, expr)
        print(f"CalcBot: The {predicted_topic} of {expr} is {solution}")
        # Show explanation after solving
        adaptive_explanation(predicted_topic)
    else:
        # No math expression: just explain the concept
        adaptive_explanation(predicted_topic)
