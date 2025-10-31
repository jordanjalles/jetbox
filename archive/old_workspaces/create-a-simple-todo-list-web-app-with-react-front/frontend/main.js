const e = React.createElement;

class TodoApp extends React.Component {
  state = { todos: [], newTitle: '' };

  componentDidMount() {
    this.fetchTodos();
  }

  fetchTodos = () => {
    fetch('http://localhost:5000/todos')
      .then(res => res.json())
      .then(todos => this.setState({ todos }))
      .catch(err => console.error(err));
  };

  addTodo = () => {
    const { newTitle } = this.state;
    if (!newTitle.trim()) return;
    fetch('http://localhost:5000/todos', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: newTitle })
    })
      .then(res => res.json())
      .then(todo => {
        this.setState(prev => ({
          todos: [...prev.todos, todo],
          newTitle: ''
        }));
      })
      .catch(err => console.error(err));
  };

  toggleComplete = id => {
    const todo = this.state.todos.find(t => t.id === id);
    if (!todo) return;
    fetch(`http://localhost:5000/todos/${id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ completed: !todo.completed })
    })
      .then(res => res.json())
      .then(updated => {
        this.setState(prev => ({
          todos: prev.todos.map(t => (t.id === id ? updated : t))
        }));
      })
      .catch(err => console.error(err));
  };

  deleteTodo = id => {
    fetch(`http://localhost:5000/todos/${id}`, { method: 'DELETE' })
      .then(res => res.json())
      .then(() => {
        this.setState(prev => ({
          todos: prev.todos.filter(t => t.id !== id)
        }));
      })
      .catch(err => console.error(err));
  };

  render() {
    const { todos, newTitle } = this.state;
    return e(
      'div',
      { id: 'app' },
      e('h1', null, 'Todo List'),
      e(
        'div',
        null,
        e('input', {
          type: 'text',
          value: newTitle,
          onChange: e => this.setState({ newTitle: e.target.value }),
          placeholder: 'New todo'
        }),
        e('button', { className: 'add-btn', onClick: this.addTodo }, 'Add')
      ),
      e('ul', { className: 'todo-list' },
        todos.map(todo =>
          e(
            'li',
            { key: todo.id, className: 'todo-item' + (todo.completed ? ' completed' : '') },
            e('span', { className: 'title', onClick: () => this.toggleComplete(todo.id) }, todo.title),
            e('button', { onClick: () => this.deleteTodo(todo.id) }, 'Delete')
          )
        )
      )
    );
  }
}

ReactDOM.createRoot(document.getElementById('root')).render(e(TodoApp));
