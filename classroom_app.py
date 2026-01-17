"""Streamlit app for running Habermas Machine deliberations in the classroom.

This app provides a simple interface for professors to:
1. Collect student opinions on a question
2. Run the Habermas Machine deliberation algorithm
3. Display consensus statements and rankings
4. Optionally run critique rounds

Usage:
    streamlit run classroom_app.py
"""

import streamlit as st
from habermas_machine import machine, types
from habermas_machine.social_choice import utils as sc_utils

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
            "gemini-2.0-flash-exp",
            "gemini-1.5-flash",
            "gemini-1.5-pro"
        ],
        help="Recommended: gemini-2.0-flash-exp for best compatibility"
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
        3. **Add participant opinions** - paste or type them in
        4. **Click "Run Opinion Round"** to generate consensus statements
        5. **Optional**: Run a critique round for refinement

        **Tips**:
        - Collect opinions beforehand (Google Form, discussion, etc.)
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
st.markdown("Enter opinions from each participant. Add or remove participants as needed.")

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
