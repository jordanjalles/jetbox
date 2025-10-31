const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(bodyParser.json());

let todos = [];
let nextId = 1;

// Get all todos
app.get('/api/todos', (req, res) => {
  res.json(todos);
});

// Create a new todo
app.post('/api/todos', (req, res) => {
  const { title, completed = false } = req.body;
  if (!title) {
    return res.status(400).json({ error: 'Title is required' });
  }
  const todo = { id: nextId++, title, completed };
  todos.push(todo);
  res.status(201).json(todo);
});

// Update a todo
app.put('/api/todos/:id', (req, res) => {
  const id = parseInt(req.params.id, 10);
  const { title, completed } = req.body;
  const todo = todos.find(t => t.id === id);
  if (!todo) {
    return res.status(404).json({ error: 'Todo not found' });
  }
  if (title !== undefined) todo.title = title;
  if (completed !== undefined) todo.completed = completed;
  res.json(todo);
});

// Delete a todo
app.delete('/api/todos/:id', (req, res) => {
  const id = parseInt(req.params.id, 10);
  const index = todos.findIndex(t => t.id === id);
  if (index === -1) {
    return res.status(404).json({ error: 'Todo not found' });
  }
  const deleted = todos.splice(index, 1)[0];
  res.json(deleted);
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
