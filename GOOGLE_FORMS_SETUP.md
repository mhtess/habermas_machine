# Google Forms + Sheets Integration Guide

This guide shows you how to seamlessly collect student opinions using Google Forms and automatically import them into the Habermas Machine app.

## Quick Setup (5 minutes)

### Step 1: Install Additional Dependencies

```bash
pip install pandas requests
```

### Step 2: Create a Google Form

1. Go to [Google Forms](https://forms.google.com)
2. Click **+ Blank** to create a new form
3. Add your form title (e.g., "Class Deliberation - Week 5")

### Step 3: Add Form Questions

Add these two questions to your form:

**Question 1:** (Optional)
- Question: "What is the discussion question?"
- Type: Short answer
- This helps you track which topic this is for

**Question 2:** (Required)
- Question: Your actual deliberation question (e.g., "Should universities require in-person attendance?")
- Type: Paragraph
- Mark as "Required"

**Example form:**
```
Question 1: What is the discussion question?
Answer: [Short answer field]

Question 2: Should universities require in-person attendance? Please explain your view in 2-3 sentences.
Answer: [Paragraph field - REQUIRED]
```

### Step 4: Link Form to Google Sheets

1. In your Google Form, click the **Responses** tab
2. Click the green **Google Sheets** icon (top right)
3. Choose **Create a new spreadsheet**
4. Click **Create**
5. Google Sheets will open automatically

### Step 5: Make the Sheet Public (Viewable)

This is **crucial** for the app to access it:

1. In the Google Sheet, click **Share** (top right)
2. Click **Change to anyone with the link**
3. Set permissions to **Viewer**
4. Click **Done**
5. Copy the sheet URL (you'll need this for the app)

### Step 6: Share Form with Students

1. In Google Forms, click **Send** (top right)
2. Copy the form link
3. Share with students via email, LMS, etc.

## Using in the Classroom App

### Before Class:

1. Share the Google Form link with students 24-48 hours before class
2. Students submit their responses
3. Responses automatically save to your Google Sheet

### During Class:

1. Open the classroom app: `streamlit run classroom_app.py`
2. Enter your API key
3. Click **"Import from Google Sheets"** (expand the section)
4. Paste your Google Sheet URL
5. Set columns:
   - **Opinion Column**: `C` (where the main responses are - column C is typically the second question)
   - **Question Column**: Leave as `A` or `B` depending on your form structure
6. Click **"Import from Sheet"**
7. Verify opinions loaded correctly
8. Click **"Run Opinion Round"**

## Google Sheets Column Layout

After students submit responses, your sheet will typically look like this:

| A (Timestamp) | B (Question 1) | C (Question 2 - Opinions) |
|---------------|----------------|---------------------------|
| 1/15/25 10:23 | Should universities... | I believe attendance should be required because... |
| 1/15/25 10:25 | Should universities... | No, students are adults and should... |
| 1/15/25 10:28 | Should universities... | It depends on the course type... |

**Import Settings:**
- Opinion Column: **C** (the main paragraph responses)
- Question Column: **B** (optional - the question text)

## Tips for Best Results

### Form Design:
- **Keep it simple**: 1-2 questions max
- **Be specific**: Clear, focused questions work best
- **Require responses**: Make the opinion question required
- **Set a deadline**: Give students 24-48 hours before class

### Question Prompts:
Good prompts help students give thoughtful responses:

```
"Please share your view on [question] in 2-3 sentences.
Explain your reasoning and any important considerations."
```

### Sample Forms:

**Ethics Course:**
```
Question: "Is it ever morally acceptable to lie to protect someone's feelings?"
Instructions: Please explain your position in 2-3 sentences, including your reasoning.
```

**Policy Course:**
```
Question: "Should the university adopt a pass/fail grading system?"
Instructions: Share your perspective and the key factors that inform your view (2-3 sentences).
```

## Troubleshooting

### "Error importing from Google Sheets"

**Solution 1**: Check sheet permissions
- Open your Google Sheet
- Click Share → Make sure it says "Anyone with the link" can view

**Solution 2**: Verify the URL
- The URL should look like: `https://docs.google.com/spreadsheets/d/LONG_ID_HERE/edit`
- Copy the entire URL from your browser

**Solution 3**: Check column letters
- Open your Google Sheet
- Count which column has the opinions (A, B, C, etc.)
- Update "Opinion Column" in the app to match

### "No opinions found in the sheet"

- Make sure students have submitted responses
- Check that "Opinion Column" matches where responses actually are
- Verify you're not using the Form URL (must be the Sheet URL)

### "Dependencies not installed"

Install the required packages:
```bash
pip install pandas requests
```

## Advanced: Multiple Rounds

For critique rounds, you can create a second Google Form:

1. Show students the winning consensus statement from Round 1
2. Create a new form asking for critiques: "What would you change about this statement?"
3. Repeat the import process for the critique round

## Privacy Considerations

- **Student names**: Form responses include timestamps by default, but you can make forms anonymous
- **To make anonymous**: In Form settings, uncheck "Collect email addresses"
- **FERPA compliance**: Don't share the Sheet URL publicly - only use it in the app during class

## Example Workflow (Complete)

**Wednesday (2 days before class):**
1. Create Google Form with deliberation question
2. Link to Google Sheet
3. Share form link with students via Canvas/email

**Thursday-Friday:**
- Students submit responses (aim for 8-15 responses)

**Friday (class day):**
1. Open classroom app
2. Import from Google Sheet
3. Run deliberation live in class
4. Discuss results

**Optional - Following week:**
1. Create critique form with previous winning statement
2. Collect critiques
3. Run critique round in next class

## Cost Note

Google Forms and Sheets are **free** for educational use. The only cost is the Gemini API (~$0.30-0.50 per session).

---

**Questions?** Check the main [CLASSROOM_SETUP.md](CLASSROOM_SETUP.md) for more teaching tips.
