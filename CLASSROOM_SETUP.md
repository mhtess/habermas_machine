# Habermas Machine - Classroom Setup Guide

This guide will help you set up and run the Habermas Machine for in-class deliberation exercises.

## Quick Start (5 minutes)

### 1. Install Requirements

You need Python 3.10 or higher. Open a terminal and run:

```bash
# Navigate to the habermas_machine folder
cd habermas_machine

# Install the package and Streamlit
pip install -e .
pip install streamlit
```

### 2. Get a Google AI Studio API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click "Create API Key"
3. Copy the key (you'll paste it into the app)

**Note**: The API is very cheap - a typical classroom session costs less than $0.50.

### 3. Run the App

```bash
streamlit run classroom_app.py
```

This will open a web browser automatically. If not, it will show you a URL like `http://localhost:8501` - open that in your browser.

### 4. Use in Class

1. **Before class**: Collect student opinions via Google Form, discussion board, or in-class discussion
2. **During class**:
   - Project the app on the screen
   - Paste in the question and opinions
   - Click "Run Opinion Round"
   - Discuss the results with students
3. **Optional**: Run a critique round for deeper engagement

## Detailed Instructions

### Preparing for Class

**Collect Opinions Beforehand** (Recommended):
- Create a Google Form with the discussion question
- Have students submit 1-2 paragraph responses before class
- Copy/paste responses into the app during class

**Or Collect Live**:
- Ask the question in class
- Have 5-10 students share their views
- Type or have a TA type them into the app

### Running a Session

1. **Enter API Key** (first time only):
   - Paste your Google AI Studio API key in the sidebar
   - The app will remember it for the session

2. **Configure Settings** (sidebar):
   - **Model**: Use `gemini-2.0-flash-exp` (recommended)
   - **Number of Candidates**: 4 works well (generates 4 alternative consensus statements)
   - **Retries**: 5 is good (handles occasional formatting issues)

3. **Enter the Question**:
   - Be specific and focused
   - Example: "Should AI tools like ChatGPT be allowed in homework assignments?"

4. **Add Opinions**:
   - Adjust "Number of Participants" to match your group
   - Paste or type each person's opinion
   - Minimum 2 participants, 5-10 works best

5. **Run Opinion Round**:
   - Click the "Run Opinion Round" button
   - Wait 30-60 seconds (progress spinner will show)
   - Review the winning consensus statement with the class

6. **Optional - Critique Round**:
   - Have students critique the winning statement
   - Paste critiques into the app
   - Click "Run Critique Round"
   - Compare the refined statement to the original

### Teaching Tips

**Discussion Questions That Work Well**:
- Current campus issues (attendance policies, grading, etc.)
- Ethics questions from your curriculum
- Policy debates where reasonable people disagree
- "Should we..." or "How should we..." questions

**How to Use Results**:
- **Compare** the consensus to individual opinions - what changed?
- **Analyze** the ranking - why did some statements win over others?
- **Critique** the process - what did the AI miss? What assumptions did it make?
- **Discuss** social choice theory - how do we aggregate preferences fairly?

**Pedagogical Approaches**:

1. **One Big Assembly** (End of Course):
   - Run one major deliberation on a course-synthesizing question
   - Show how diverse views can find common ground
   - Reflect on the semester's learning

2. **Ongoing Use**:
   - Weekly deliberations on reading topics
   - Start each class with a mini-deliberation
   - Track how class consensus evolves over the semester

3. **Meta-Analysis**:
   - Critique the AI's consensus
   - Compare AI-mediated vs. traditional deliberation
   - Discuss algorithmic social choice

## Troubleshooting

### "Please enter your API key"
- Make sure you've pasted your Google AI Studio API key in the sidebar
- Get one at: https://aistudio.google.com/app/apikey

### "Retrying with new seed"
- This is normal! The model occasionally returns incorrectly formatted responses
- The app will automatically retry (up to 5 times by default)
- If it keeps failing, try using `gemini-1.5-pro` instead

### App won't start
- Check that you have Python 3.10+ installed: `python --version`
- Make sure you installed the package: `pip install -e .`
- Make sure you installed streamlit: `pip install streamlit`

### API Costs
- Gemini Flash is very cheap: ~$0.30-0.50 per classroom session
- Google provides free credits for new accounts
- Set up billing alerts if concerned

### Need Help?
- Check the [main README](README.md) for more details
- File an issue on GitHub: https://github.com/google-deepmind/habermas_machine/issues

## Example Workflow

Here's a complete example for a 50-minute class:

**Before Class (15 minutes)**:
1. Create Google Form with question: "Should the university require all students to take an ethics course?"
2. Share with students 24 hours before class
3. Collect 8-10 responses

**During Class**:
1. **Setup (2 minutes)**:
   - Open classroom_app.py
   - Paste API key (if first time)
   - Paste question

2. **Input Opinions (3 minutes)**:
   - Copy/paste student responses from Google Form
   - Set to 8-10 participants

3. **Run & Discuss (15 minutes)**:
   - Click "Run Opinion Round"
   - While running: explain the algorithm
   - Show winning statement on screen
   - Discuss: "What do you notice? What's missing?"

4. **Critique Round (15 minutes)**:
   - Collect verbal critiques from 5-6 students
   - Type them into the app
   - Run critique round
   - Compare original vs. refined statement

5. **Reflection (15 minutes)**:
   - Discuss the process
   - Compare to traditional deliberation
   - Talk about algorithmic fairness

## Advanced: Hosting Online

If you want students to access the app remotely instead of just projecting it:

1. Create a [Streamlit Cloud](https://streamlit.io/cloud) account (free)
2. Fork the repository to your GitHub
3. Deploy `classroom_app.py` on Streamlit Cloud
4. Share the URL with students

This allows asynchronous use between classes.

## Citation

If you use this in research or publications:

```
Tessler, M. H., Bakker, M. A., Jarrett, D., Sheahan, H., Chadwick, M. J.,
Koster, R., Evans, G., Campbell-Gillingham, J., Collins, T., Parkes, D. C.,
Botvinick, M., and Summerfield, C. (2024). AI can help humans find common
ground in democratic deliberation. Science, 385(6714), eadq2852.
```

---

**Happy deliberating! Questions? Email the professor or file a GitHub issue.**
