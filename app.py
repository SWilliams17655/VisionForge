from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def home_page():
    return "Hello World"

if __name__ == "__main__":
    app.run(host="0.0.0.0")
