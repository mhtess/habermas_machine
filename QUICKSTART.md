# Habermas Machine - Quick Start (2 Minutes)

## For Someone Setting This Up

Here's everything you need to get the classroom app running:

### Step 1: Install Python Dependencies

```bash
# Navigate to the project folder
cd habermas_machine

# Install the habermas machine package
pip install -e .

# Install Streamlit (the web framework)
pip install streamlit
```

### Step 2: Get API Key

1. Go to https://aistudio.google.com/app/apikey
2. Click "Create API Key"
3. Copy the key

### Step 3: Run the App

```bash
streamlit run classroom_app.py
```

A browser window will open automatically at http://localhost:8501

### Step 4: Use It

1. Paste the API key in the sidebar
2. Enter a question
3. Add participant opinions (2+ people)
4. Click "Run Opinion Round"
5. Wait ~30 seconds
6. View the consensus statement

**That's it!**

---

## What This Does

The Habermas Machine takes diverse opinions on a question and uses AI to find common ground:

1. **Input**: A question + 5-10 people's opinions
2. **Process**: AI generates multiple consensus statements, predicts how each person would rank them, and uses democratic voting to pick the best
3. **Output**: A consensus statement that represents the group's collective view

---

## Cost

- Uses Google's Gemini API
- Very cheap: ~$0.30-0.50 per classroom session
- Google gives free credits to new users

---

## Troubleshooting

**"Module not found"**: Run `pip install -e .` from the habermas_machine folder

**"API key invalid"**: Make sure you copied the entire key from AI Studio

**App crashes during deliberation**: Increase "Retries on Error" to 10 in the sidebar

**Need help?**: See [CLASSROOM_SETUP.md](CLASSROOM_SETUP.md) for detailed instructions

---

## For the Professor

Once it's running:
- Collect student opinions beforehand (Google Form works great)
- Project the app during class
- Paste in opinions and run the deliberation live
- Discuss results with students

See CLASSROOM_SETUP.md for teaching tips and example workflows.
