#!/bin/bash
# Test orchestrator with live server workflow
# This script simulates user interaction with the orchestrator

echo "Testing Orchestrator with Live Server Workflow"
echo "==============================================="
echo ""

# Task 1: Create web app and start server
echo "TASK 1: Create simple web app with homepage and start HTTP server"
echo ""
python orchestrator_main.py "Create a simple web app in a new directory. It should have index.html with 'Welcome to My App' as the heading, and style.css with a blue background. After creating the files, start an HTTP server on port 8123 using start_server tool with name 'myapp'."

echo ""
echo "Waiting 3 seconds for server to fully start..."
sleep 3

# Check if server is running
echo ""
echo "Checking if server is running..."
curl -s http://localhost:8123 | head -5 || echo "Server not accessible yet"

echo ""
echo "==============================================="
echo ""

# Task 2: Add a feature (server should still be running)
echo "TASK 2: Add contact page (server should still be running)"
echo ""
python orchestrator_main.py "Add a contact page to the web app. Create contact.html with 'Contact Us' heading. Use check_server tool to verify the server is still running before you start."

echo ""
echo "Waiting 2 seconds..."
sleep 2

# Verify server is still running and serving new content
echo ""
echo "Verifying server is still serving content..."
curl -s http://localhost:8123/contact.html | head -5 || echo "Contact page not accessible"

echo ""
echo "==============================================="
echo ""
echo "Test complete! The server should have persisted across both tasks."
echo "Now we'll start orchestrator in interactive mode to manually verify shutdown."
echo ""
