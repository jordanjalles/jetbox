# Implementation of TodoList
import json
import os

class TodoList:
    def __init__(self):
        self.tasks = []  # list of dicts with keys: description, completed

    def add_task(self, description: str):
        self.tasks.append({'description': description, 'completed': False})

    def remove_task(self, description: str):
        self.tasks = [t for t in self.tasks if t['description'] != description]

    def mark_complete(self, description: str):
        for t in self.tasks:
            if t['description'] == description:
                t['completed'] = True
                break

    def list_pending(self):
        return [t['description'] for t in self.tasks if not t['completed']]

    def save_to_file(self, filename: str):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.tasks, f, ensure_ascii=False, indent=2)

    def load_from_file(self, filename: str):
        if not os.path.exists(filename):
            return
        with open(filename, 'r', encoding='utf-8') as f:
            self.tasks = json.load(f)
