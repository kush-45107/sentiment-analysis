from flask import Flask, render_template, request
import joblib

model = joblib.load("sentiment_analysis_model.pkl")
tfidf = joblib.load("tfidfvector.pkl")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        user_text = request.form["text"]
        if user_text.strip() != "":
            # Transform input
            text_vec = tfidf.transform([user_text])
            pred = model.predict(text_vec)[0]
            prediction = f"Predicted Sentiment: {pred}"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
