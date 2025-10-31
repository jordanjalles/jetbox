from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# In-memory todo list
# Each todo is a dict with id and text
todos = []
next_id = 1

@app.route("/")
def index():
    return render_template("index.html", todos=todos)

@app.route("/add", methods=["POST"])
def add():
    global next_id
    text = request.form.get("text")
    if text:
        todos.append({"id": next_id, "text": text})
        next_id += 1
    return redirect(url_for("index"))

@app.route("/delete/<int:todo_id>", methods=["POST"])
def delete(todo_id):
    global todos
    todos = [t for t in todos if t["id"] != todo_id]
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
