# Smart Home Agent - Usage Guide

## ⚠️ IMPORTANT SAFETY NOTES

The Smart Home agent controls **real physical devices** in your home. Always:
- Provide explicit, clear commands
- Double-check team selection
- Never run without a specific goal

## Correct Usage

### Method 1: Direct Team Selection (Recommended)

```bash
# Specify smart_home team explicitly
python agent.py --team smart_home "Turn Jordan's office lights to 75%"
python agent.py --team smart_home "Turn off all lights"
python agent.py --team smart_home "What's the temperature?"
```

### Method 2: Interactive Selection

```bash
# Run without team flag
python agent.py "Turn on the living room lights"

# Select from menu:
# 5. Smart Home Team [CONTROLS PHYSICAL DEVICES]
```

### Method 3: Quick Runner Script

```bash
# Use the dedicated script
python run_smart_home.py "Turn Jordan's office lights to 100%"
python run_smart_home.py "List all devices"
```

## ❌ Common Mistakes

### DON'T: Run smart home without a goal
```bash
python agent.py --team smart_home
# This is dangerous - agent has no instructions!
```

### DON'T: Use wrong flags
```bash
python agent.py --chatbot  # Wrong flag, does nothing
python agent.py --chat     # This is for chatbot mode, not smart home
```

### DON'T: Select wrong team
```
python agent.py "hello"
# Then accidentally select: 5. Smart Home Team
# This could trigger unwanted device control!
```

## Interactive Mode (Chat)

**Smart Home agent does NOT support interactive chat mode.**

It's designed for:
- ✅ Single explicit commands
- ✅ Scheduled automation tasks
- ❌ NOT for ongoing conversation

For chat, use:
```bash
python agent.py --team chatbot "hello"
# Or select: 1. Simple Chatbot Team
```

## Scheduled Automation

To run smart home tasks automatically:

1. Edit `config/agents/smart_home_controller.yaml`:
   ```yaml
   deployment:
     enabled: true
     schedule: "*/15 * * * *"  # Every 15 minutes
     goal: "Turn off lights if nobody home after 10pm"
   ```

2. Start the agent manager:
   ```bash
   python jetbox.py start --foreground
   ```

3. Monitor:
   ```bash
   python jetbox.py status
   python jetbox.py logs smart_home_controller --follow
   ```

## Credentials

Credentials are stored securely in `.jetbox/secrets/home_assistant.json`:
```json
{
  "url": "http://192.168.50.4:8123",
  "token": "your_token_here"
}
```

This file is gitignored and auto-loads when the smart home agent starts.

## Troubleshooting

**Agent won't connect:**
- Check `.jetbox/secrets/home_assistant.json` exists
- Verify URL and token are correct
- Test: `curl -H "Authorization: Bearer TOKEN" URL/api/`

**Agent takes unexpected actions:**
- Always provide EXPLICIT commands
- Avoid vague goals like "help" or "chat"
- Check the team selection (should say "[CONTROLS PHYSICAL DEVICES]")

**Want to undo an action:**
```bash
# Just run the opposite command
python agent.py --team smart_home "Turn lights back on"
```

## Safety Features

The agent now has multiple safety checks:
1. ✅ Team selection shows warning: "[CONTROLS PHYSICAL DEVICES]"
2. ✅ Agent refuses to act on unclear/missing goals
3. ✅ Conservative behavior - only acts on explicit commands
4. ✅ Always verifies state before and after actions

## Examples

```bash
# Lights
python agent.py --team smart_home "Turn on bedroom lights to 50%"
python agent.py --team smart_home "Turn off all lights in living room"

# Climate
python agent.py --team smart_home "Set thermostat to 72 degrees"
python agent.py --team smart_home "What's the current temperature?"

# Status
python agent.py --team smart_home "List all my devices"
python agent.py --team smart_home "Show me which lights are currently on"

# Automations
python agent.py --team smart_home "Run my bedtime automation"
python agent.py --team smart_home "List all automations"
```

## Need Help?

- **Quick Start**: See `docs/SPECIALIZED_AGENTS_QUICKSTART.md`
- **Creating Custom Agents**: See `docs/SPECIALIZED_AGENTS_DEV.md`
- **Home Assistant Setup**: See your HA instance documentation
