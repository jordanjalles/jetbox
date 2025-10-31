import React, { useEffect, useState } from 'react';
import './App.css';

const API_URL = 'http://localhost:5000/api/todos';

function App() {
  const [todos, setTodos] = useState([]);
  const [newTitle, setNewTitle] = useState('');

  const fetchTodos = async () => {
    const res = await fetch(API_URL);
    const data = await res.json();
    setTodos(data);
  };

  useEffect(() => {
    fetchTodos();
  }, []);

  const addTodo = async () => {
    if (!newTitle.trim()) return;
    const res = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: newTitle }),
    });
    const created = await res.json();
    setTodos([...todos, created]);
    setNewTitle('');
  };

  const toggleComplete = async (id, completed) => {
    const res = await fetch(`${API_URL}/${id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ completed: !completed }),
    });
    const updated = await res.json();
    setTodos(todos.map(t => (t.id === id ? updated : t)));
  };

  const deleteTodo = async id => {
    await fetch(`${API_URL}/${id}`, { method: 'DELETE' });
    setTodos(todos.filter(t => t.id !== id));
  };

  return (
    <div className="App">
      <h1>Todo List</h1>
      <div className="input-group">
        <input
          type="text"
          value={newTitle}
          onChange={e => setNewTitle(e.target.value)}
          placeholder="New todo"
        />
        <button onClick={addTodo}>Add</button>
      </div>
      <ul>
        {todos.map(todo => (
          <li key={todo.id} className={todo.completed ? 'completed' : ''}>
            <input
              type="checkbox"
              checked={todo.completed}
              onChange={() => toggleComplete(todo.id, todo.completed)}
            />
            {todo.title}
            <button onClick={() => deleteTodo(todo.id)}>Delete</button>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default App;
