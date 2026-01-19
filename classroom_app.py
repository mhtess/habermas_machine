"""Streamlit app for running Habermas Machine deliberations in the classroom.

This app provides a simple interface for professors to:
1. Collect student opinions on a question
2. Run the Habermas Machine deliberation algorithm
3. Display consensus statements and rankings
4. Optionally run critique rounds

Usage:
    streamlit run classroom_app.py
"""

import re
import streamlit as st
from habermas_machine import machine, types
from habermas_machine.social_choice import utils as sc_utils

# Optional Google Sheets integration
try:
    import pandas as pd
    import requests
    SHEETS_AVAILABLE = True
except ImportError:
    SHEETS_AVAILABLE = False


def fetch_from_google_sheets(sheet_url: str, opinion_column: str = "B", question_column: str = "A") -> tuple[str | None, list[str]]:
    """Fetch opinions from a public Google Sheet.

    Args:
        sheet_url: URL of the Google Sheet
        opinion_column: Column letter containing opinions (default: B)
        question_column: Column letter for the question (optional, default: A)

    Returns:
        Tuple of (question, list of opinions)
    """
    if not SHEETS_AVAILABLE:
        raise ImportError("Google Sheets integration requires: pip install pandas requests")

    try:
        import pandas as pd
        import io
        import requests

        # Extract sheet ID from URL
        match = re.search(r'/spreadsheets/d/([a-zA-Z0-9-_]+)', sheet_url)
        if not match:
            raise ValueError("Invalid Google Sheets URL")

        sheet_id = match.group(1)

        # For public sheets, export as CSV (no authentication needed)
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"

        response = requests.get(url)
        response.raise_for_status()

        df = pd.read_csv(io.StringIO(response.text))

        # Get opinions from specified column (skip header row)
        col_idx = ord(opinion_column.upper()) - ord('A')
        if col_idx >= len(df.columns):
            raise ValueError(f"Column {opinion_column} not found in sheet")

        opinions = []
        # pandas read_csv already used first row as headers, so start from row 0
        start_row = 0

        # Debug: print what we're reading
        print(f"DEBUG: DataFrame has {len(df)} rows, {len(df.columns)} columns")
        print(f"DEBUG: Column names: {df.columns.tolist()}")
        print(f"DEBUG: Looking for opinions in column index {col_idx}")

        for i in range(start_row, len(df)):
            opinion = str(df.iloc[i, col_idx]).strip()
            print(f"DEBUG: Row {i}, opinion value: '{opinion}'")
            if opinion and opinion != 'nan':
                opinions.append(opinion)

        print(f"DEBUG: Found {len(opinions)} opinions")

        # Get question from the first row if available
        question = None
        if question_column and len(df.columns) > 0 and len(df) > 0:
            col_idx = ord(question_column.upper()) - ord('A')
            if col_idx < len(df.columns):
                question = str(df.iloc[0, col_idx]).strip()
                # Check if it's actually a question (not empty or 'nan')
                if question == 'nan' or not question:
                    question = None

        return question, opinions

    except Exception as e:
        raise Exception(f"Error fetching from Google Sheets: {str(e)}")

# Page configuration
st.set_page_config(
    page_title="Habermas Machine - Classroom Demo",
    page_icon="🗳️",
    layout="wide"
)

# Title and description
st.title("🗳️ Habermas Machine - Democratic Deliberation")
st.markdown("""
This tool helps groups find common ground through AI-mediated deliberation.
Enter a question, collect opinions from participants, and watch the machine
generate consensus statements.
""")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # API Key input
    api_key = st.text_input(
        "Google AI Studio API Key",
        type="password",
        help="Get your API key from https://aistudio.google.com/app/apikey"
    )

    if api_key:
        import os
        os.environ['GOOGLE_API_KEY'] = api_key

    # Model selection
    model_name = st.selectbox(
        "Gemini Model",
        options=[
            "gemini-flash-latest",
            "gemini-pro-latest",
        ],
        help="Recommended: gemini-flash-latest for best compatibility"
    )

    # Number of candidates
    num_candidates = st.slider(
        "Number of Candidate Statements",
        min_value=2,
        max_value=8,
        value=4,
        help="How many alternative consensus statements to generate"
    )

    # Number of retries
    num_retries = st.slider(
        "Retries on Error",
        min_value=1,
        max_value=10,
        value=5,
        help="How many times to retry if the model returns an incorrect format"
    )

    st.divider()

    # Instructions
    with st.expander("📖 How to Use"):
        st.markdown("""
        1. **Enter API Key** (sidebar) - Get it from [AI Studio](https://aistudio.google.com/app/apikey)
        2. **Enter your question** below
        3. **Add participant opinions**:
           - Option A: Import from Google Sheets (recommended for Google Forms)
           - Option B: Paste or type them in manually
        4. **Click "Run Opinion Round"** to generate consensus statements
        5. **Optional**: Run a critique round for refinement

        **Tips**:
        - Collect opinions beforehand via Google Form → auto-saves to Sheets
        - 3-10 participants works well
        - Clear, specific questions get better results
        """)

    with st.expander("💡 Example Questions"):
        st.markdown("""
        - Should the university require in-person attendance?
        - How should AI tools be used in coursework?
        - What is the most important factor in moral decision-making?
        - Should we prioritize individual rights or collective welfare?
        """)

# Main content area
if not api_key:
    st.warning("⚠️ Please enter your Google AI Studio API key in the sidebar to begin.")
    st.stop()

# Initialize session state
if 'opinions' not in st.session_state:
    st.session_state.opinions = [""] * 3
if 'critiques' not in st.session_state:
    st.session_state.critiques = []
if 'winner' not in st.session_state:
    st.session_state.winner = None
if 'sorted_statements' not in st.session_state:
    st.session_state.sorted_statements = None
if 'hm' not in st.session_state:
    st.session_state.hm = None

# Question input
question = st.text_area(
    "Discussion Question",
    placeholder="e.g., Should the government provide universal free childcare from birth?",
    height=100
)

st.divider()

# Opinion collection
st.subheader("👥 Participant Opinions")

# Google Sheets import option
if SHEETS_AVAILABLE:
    with st.expander("📊 Import from Google Sheets (Optional)"):
        st.markdown("""
        **How to use:**
        1. Create a Google Form to collect opinions
        2. Form responses save to a Google Sheet automatically
        3. Make the Sheet **publicly viewable** (Share → Anyone with link can view)
        4. Paste the Sheet URL below and click Import

        **Expected format:**
        - Column A (optional): Question in first row
        - Column B: Participant opinions (one per row, starting from row 2)
        """)

        sheet_url = st.text_input(
            "Google Sheets URL",
            placeholder="https://docs.google.com/spreadsheets/d/...",
            help="The sheet must be publicly viewable"
        )

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            opinion_column = st.text_input("Opinion Column", value="B", max_chars=1)
        with col2:
            question_column = st.text_input("Question Column (optional)", value="A", max_chars=1)

        if st.button("📥 Import from Sheet"):
            if not sheet_url:
                st.error("Please enter a Google Sheets URL")
            else:
                try:
                    with st.spinner("Fetching data from Google Sheets..."):
                        imported_question, imported_opinions = fetch_from_google_sheets(
                            sheet_url,
                            opinion_column=opinion_column,
                            question_column=question_column if question_column else None
                        )

                        if not imported_opinions:
                            st.error("No opinions found in the sheet. Check the column letter and row numbers.")
                        else:
                            # Update session state with imported data
                            st.session_state.opinions = imported_opinions

                            # Also update the question if found
                            if imported_question and imported_question != 'nan':
                                st.session_state.imported_question = imported_question

                            st.success(f"✅ Imported {len(imported_opinions)} opinions from Google Sheets!")
                            if imported_question:
                                st.info(f"💡 Detected question: {imported_question[:100]}...")

                            # Force a rerun to show the imported data
                            st.rerun()

                except Exception as e:
                    st.error(f"Error importing from Google Sheets: {str(e)}")
                    st.markdown("""
                    **Troubleshooting:**
                    - Make sure the Sheet is set to "Anyone with the link can view"
                    - Check that the column letters are correct
                    - Verify the URL is correct
                    """)
else:
    with st.expander("📊 Enable Google Sheets Import"):
        st.info("Install dependencies to enable Google Sheets import:")
        st.code("pip install pandas requests")

st.markdown("---")
st.markdown("Enter opinions from each participant. Add or remove participants as needed.")

# Use imported question if available
if 'imported_question' in st.session_state and st.session_state.imported_question:
    if not question:  # Only auto-fill if question field is empty
        st.info(f"💡 Using question from Google Sheets: {st.session_state.imported_question}")
        question = st.session_state.imported_question

col1, col2 = st.columns([3, 1])
with col2:
    num_participants = st.number_input(
        "Number of Participants",
        min_value=2,
        max_value=20,
        value=len(st.session_state.opinions),
        help="Adjust to add or remove opinion boxes"
    )

# Adjust opinions list based on participant count
if num_participants > len(st.session_state.opinions):
    st.session_state.opinions.extend([""] * (num_participants - len(st.session_state.opinions)))
elif num_participants < len(st.session_state.opinions):
    st.session_state.opinions = st.session_state.opinions[:num_participants]

# Opinion input fields
for i in range(num_participants):
    st.session_state.opinions[i] = st.text_area(
        f"Participant {i+1}",
        value=st.session_state.opinions[i],
        height=120,
        key=f"opinion_{i}"
    )

st.divider()

# Run opinion round
if st.button("🚀 Run Opinion Round", type="primary", use_container_width=True):
    # Validate inputs
    if not question.strip():
        st.error("Please enter a discussion question.")
        st.stop()

    valid_opinions = [op.strip() for op in st.session_state.opinions if op.strip()]
    if len(valid_opinions) < 2:
        st.error("Please enter at least 2 participant opinions.")
        st.stop()

    # Initialize Habermas Machine
    with st.spinner("Initializing Habermas Machine..."):
        try:
            statement_client = types.LLMCLient.AISTUDIO.get_client(model_name)
            reward_client = types.LLMCLient.AISTUDIO.get_client(model_name)

            statement_model = types.StatementModel.CHAIN_OF_THOUGHT.get_model()
            reward_model = types.RewardModel.CHAIN_OF_THOUGHT_RANKING.get_model()
            social_choice_method = types.RankAggregation.SCHULZE.get_method(
                tie_breaking_method=sc_utils.TieBreakingMethod.TBRC
            )

            hm = machine.HabermasMachine(
                question=question,
                statement_client=statement_client,
                reward_client=reward_client,
                statement_model=statement_model,
                reward_model=reward_model,
                social_choice_method=social_choice_method,
                num_candidates=num_candidates,
                num_citizens=len(valid_opinions),
                verbose=False,
                num_retries_on_error=num_retries,
            )

            st.session_state.hm = hm

        except Exception as e:
            st.error(f"Error initializing: {str(e)}")
            st.stop()

    # Run deliberation
    with st.spinner("Running deliberation... This may take 30-60 seconds."):
        try:
            winner, sorted_statements = hm.mediate(valid_opinions)
            st.session_state.winner = winner
            st.session_state.sorted_statements = sorted_statements
            st.session_state.critiques = [""] * len(valid_opinions)
            st.success("✅ Opinion round complete!")
        except Exception as e:
            st.error(f"Error running deliberation: {str(e)}")
            st.exception(e)

# Display results
if st.session_state.winner:
    st.divider()
    st.subheader("📊 Results - Opinion Round")

    # Winning statement
    st.markdown("### 🏆 Winning Consensus Statement")
    st.info(st.session_state.winner)

    # All statements
    with st.expander("📋 View All Generated Statements (Ranked)"):
        for i, stmt in enumerate(st.session_state.sorted_statements, 1):
            st.markdown(f"**Statement {i}:**")
            st.write(stmt)
            st.divider()

    # Critique round option
    st.divider()
    st.subheader("💭 Optional: Critique Round")
    st.markdown("Collect critiques of the winning statement to refine it further.")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("*Participants can critique the winning statement above. The machine will generate an improved version.*")

    # Critique inputs
    for i in range(len(valid_opinions)):
        st.session_state.critiques[i] = st.text_area(
            f"Critique from Participant {i+1}",
            value=st.session_state.critiques[i],
            height=80,
            key=f"critique_{i}"
        )

    # Run critique round
    if st.button("🔄 Run Critique Round", type="secondary", use_container_width=True):
        valid_critiques = [c.strip() for c in st.session_state.critiques if c.strip()]

        if len(valid_critiques) < 2:
            st.error("Please enter at least 2 critiques to run the critique round.")
            st.stop()

        with st.spinner("Running critique round... This may take 30-60 seconds."):
            try:
                winner, sorted_statements = st.session_state.hm.mediate(valid_critiques)

                st.divider()
                st.subheader("📊 Results - Critique Round")

                st.markdown("### 🏆 Refined Consensus Statement")
                st.success(winner)

                with st.expander("📋 View All Refined Statements (Ranked)"):
                    for i, stmt in enumerate(sorted_statements, 1):
                        st.markdown(f"**Statement {i}:**")
                        st.write(stmt)
                        st.divider()

            except Exception as e:
                st.error(f"Error running critique round: {str(e)}")
                st.exception(e)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
    Habermas Machine - Based on research by Tessler et al. (2024), Science<br>
    "AI can help humans find common ground in democratic deliberation"
</div>
""", unsafe_allow_html=True)
