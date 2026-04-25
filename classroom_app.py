"""Streamlit app for running Habermas Machine deliberations in the classroom.

This app provides a simple interface for professors to:
1. Collect student opinions on a question
2. Run the Habermas Machine deliberation algorithm
3. Display consensus statements and rankings
4. Optionally run critique rounds

Usage:
    streamlit run classroom_app.py
"""

import io
import sys
import threading
import time
import streamlit as st
from cost_estimation import CostEstimate, estimate_cost
from habermas_machine import machine, types
from habermas_machine.llm_client import aistudio_client
from habermas_machine.social_choice import utils as sc_utils

# Optional Google Sheets integration
try:
    import pandas as pd
    import requests
    from sheets_io import fetch_from_google_sheets
    SHEETS_AVAILABLE = True
except ImportError:
    SHEETS_AVAILABLE = False

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

    # Model selection. We curate the dropdown order so that long-context,
    # higher-quality "pro" variants surface first — at scale (50+ participants
    # with multi-paragraph opinions) the prompt routinely runs to 100K+ tokens
    # and the lite variant's reasoning quality degrades noticeably.
    _preferred_order = (
        'gemini-pro-latest',
        'gemini-2.5-pro',
        'gemini-flash-latest',
        'gemini-2.5-flash',
        'gemini-3-pro-preview',
        'gemini-3-flash-preview',
        'gemini-2.5-flash-lite',
    )
    _supported = set(aistudio_client.SUPPORTED_MODELS)
    _model_options = [m for m in _preferred_order if m in _supported]
    _model_options += [m for m in aistudio_client.SUPPORTED_MODELS
                       if m not in _model_options]
    model_name = st.selectbox(
        "Gemini Model",
        options=_model_options,
        help=(
            "All listed models support >=1M-token context. "
            "For small groups (<20), gemini-flash-latest is fastest. "
            "For larger groups or long opinions, prefer gemini-pro-latest."
        ),
    )

    # Number of candidates. Capped at 26 because the per-citizen ranker
    # uses single-letter labels (A-Z) in both the prompt and its arrow-notation
    # regex (see reward_model/cot_ranking_model.py). Going past Z would
    # require switching to multi-char or numeric labels and rewriting the
    # few-shot examples.
    num_candidates = st.slider(
        "Number of Candidate Statements",
        min_value=2,
        max_value=26,
        value=4,
        help=(
            "How many alternative consensus statements to generate. "
            "More candidates = more diversity, but also longer ranking "
            "prompts and noisier per-citizen rankings past ~10 items."
        ),
    )

    # Number of retries
    num_retries = st.slider(
        "Retries on Error",
        min_value=1,
        max_value=10,
        value=5,
        help="How many times to retry if the model returns an incorrect format"
    )

    # Concurrent LLM call cap. Per-citizen ranking calls are independent and
    # I/O-bound, so parallelism is a big win — but spawning one thread per
    # citizen at 100+ participants will saturate the API's RPM quota and
    # trigger 429s. 16 is a safe default for paid Gemini quotas; bump up if
    # your quota allows.
    max_concurrent = st.slider(
        "Max concurrent LLM calls",
        min_value=1,
        max_value=64,
        value=16,
        help=(
            "Caps simultaneous ranking calls to the Gemini API. "
            "Higher = faster, but increases the chance of hitting "
            "rate limits (transient errors are retried automatically)."
        ),
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
# Pending/confirmed flags for the cost-confirmation dialog. The dialog flow
# is: button click -> set pending_* -> dialog renders -> Confirm sets
# confirmed_* -> the run executes on the next rerun.
if 'pending_opinion_run' not in st.session_state:
    st.session_state.pending_opinion_run = None
if 'confirmed_opinion_run' not in st.session_state:
    st.session_state.confirmed_opinion_run = None
if 'pending_critique_run' not in st.session_state:
    st.session_state.pending_critique_run = None
if 'confirmed_critique_run' not in st.session_state:
    st.session_state.confirmed_critique_run = None


def _render_cost_body(estimate: CostEstimate, num_participants_in_run: int):
    """Renders the cost summary used inside the confirmation dialog."""
    st.markdown(
        f"### Estimated cost: "
        f"**${estimate.cost_low_usd:.2f} – ${estimate.cost_high_usd:.2f}**"
    )
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Participants", num_participants_in_run)
    col_b.metric("LLM calls", f"~{estimate.num_llm_calls}")
    col_c.metric(
        "Tokens (in / out)",
        f"~{estimate.input_tokens // 1000}K / ~{estimate.output_tokens // 1000}K",
    )
    st.caption(
        "Rough estimate based on prompt size and Gemini list pricing. "
        "Actual cost depends on model behavior and current pricing — "
        "treat the range as a ballpark, not a quote."
    )
    if not estimate.pricing_known:
        st.warning(
            "No pricing data on file for the selected model — falling back "
            "to Gemini Pro list rates, which may overestimate the cost."
        )
    if estimate.is_critique:
        st.info(
            "Critique rounds re-run the full pipeline with critiques + the "
            "previous winner added to the prompt, so they're slightly more "
            "expensive than the opinion round."
        )


@st.dialog("Confirm deliberation cost", width="large")
def _confirm_opinion_dialog(estimate: CostEstimate, num_participants_in_run: int):
    _render_cost_body(estimate, num_participants_in_run)
    col1, col2 = st.columns(2)
    if col1.button("✅ Confirm and run", type="primary", use_container_width=True,
                   key="confirm_opinion_btn"):
        st.session_state.confirmed_opinion_run = True
        st.session_state.pending_opinion_run = None
        st.rerun()
    if col2.button("❌ Cancel", use_container_width=True,
                   key="cancel_opinion_btn"):
        st.session_state.pending_opinion_run = None
        st.rerun()


@st.dialog("Confirm critique-round cost", width="large")
def _confirm_critique_dialog(estimate: CostEstimate, num_participants_in_run: int):
    _render_cost_body(estimate, num_participants_in_run)
    col1, col2 = st.columns(2)
    if col1.button("✅ Confirm and run", type="primary", use_container_width=True,
                   key="confirm_critique_btn"):
        st.session_state.confirmed_critique_run = True
        st.session_state.pending_critique_run = None
        st.rerun()
    if col2.button("❌ Cancel", use_container_width=True,
                   key="cancel_critique_btn"):
        st.session_state.pending_critique_run = None
        st.rerun()

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

                            # Also set individual widget states (since text areas use keys)
                            for i, op in enumerate(imported_opinions):
                                st.session_state[f"opinion_{i}"] = op

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
        max_value=200,
        value=len(st.session_state.opinions),
        help=(
            "Adjust to add or remove opinion boxes. "
            "For >40 participants, use a long-context model "
            "(e.g. gemini-2.5-pro / gemini-pro-latest) and expect "
            "deliberation to take several minutes."
        ),
    )

# Adjust opinions list based on participant count
if num_participants > len(st.session_state.opinions):
    st.session_state.opinions.extend([""] * (num_participants - len(st.session_state.opinions)))
elif num_participants < len(st.session_state.opinions):
    st.session_state.opinions = st.session_state.opinions[:num_participants]

# Opinion input fields. We paginate above ~25 participants so Streamlit
# doesn't have to re-render hundreds of text areas on every interaction —
# session_state persists widget values for all participants regardless of
# which page is currently visible.
OPINION_PAGE_SIZE = 25

# Seed widget state for every participant so unrendered pages keep their data.
for i in range(num_participants):
    if f"opinion_{i}" not in st.session_state:
        st.session_state[f"opinion_{i}"] = st.session_state.opinions[i]

if num_participants > OPINION_PAGE_SIZE:
    num_pages = (num_participants - 1) // OPINION_PAGE_SIZE + 1
    page = st.number_input(
        f"Page (showing {OPINION_PAGE_SIZE} participants per page)",
        min_value=1,
        max_value=num_pages,
        value=1,
        key="opinion_page",
    )
    start_idx = (page - 1) * OPINION_PAGE_SIZE
    end_idx = min(start_idx + OPINION_PAGE_SIZE, num_participants)
    st.caption(
        f"Showing participants {start_idx + 1}–{end_idx} "
        f"of {num_participants}"
    )
else:
    start_idx, end_idx = 0, num_participants

for i in range(start_idx, end_idx):
    st.text_area(
        f"Participant {i + 1}",
        height=120,
        key=f"opinion_{i}",
    )

# Sync widget state back to the opinions list for ALL participants — not
# just the visible page — so the canonical list stays in sync as the user
# pages through.
for i in range(num_participants):
    if f"opinion_{i}" in st.session_state:
        st.session_state.opinions[i] = st.session_state[f"opinion_{i}"]

st.divider()

# Run opinion round (gated behind a cost-estimate confirmation dialog).
opinion_btn_clicked = st.button(
    "🚀 Run Opinion Round", type="primary", use_container_width=True
)

if opinion_btn_clicked:
    # Validate before opening the dialog so we don't spend a click on
    # something that's going to fail validation anyway.
    if not question.strip():
        st.error("Please enter a discussion question.")
    else:
        _valid_opinions = [
            op.strip() for op in st.session_state.opinions if op.strip()
        ]
        if len(_valid_opinions) < 2:
            st.error("Please enter at least 2 participant opinions.")
        else:
            _estimate = estimate_cost(
                question=question,
                opinions=_valid_opinions,
                num_candidates=num_candidates,
                model_name=model_name,
            )
            st.session_state.pending_opinion_run = {
                'estimate': _estimate,
                'num_participants_in_run': len(_valid_opinions),
            }

if st.session_state.pending_opinion_run is not None:
    _pending = st.session_state.pending_opinion_run
    _confirm_opinion_dialog(
        _pending['estimate'], _pending['num_participants_in_run']
    )

if st.session_state.confirmed_opinion_run:
    st.session_state.confirmed_opinion_run = None
    valid_opinions = [
        op.strip() for op in st.session_state.opinions if op.strip()
    ]

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
                verbose=True,  # Enable logging to terminal
                num_retries_on_error=num_retries,
                max_workers=min(max_concurrent, len(valid_opinions)),
            )

            print("\n" + "="*80)
            print(f"STARTING DELIBERATION: {question}")
            print(f"Participants: {len(valid_opinions)}, Candidates: {num_candidates}")
            print("="*80 + "\n")

            st.session_state.hm = hm

        except Exception as e:
            st.error(f"Error initializing: {str(e)}")
            st.stop()

    # Run deliberation with progress tracking
    try:
        with st.status("Running deliberation...", expanded=True) as status:
            status.write("🔄 Generating candidate statements...")

            # Capture stdout to monitor progress
            output_buffer = io.StringIO()
            result_container = [None, None]  # [winner, sorted_statements]
            exception_container = [None]

            def run_mediate():
                """Run mediation in thread with output capture."""
                old_stdout = sys.stdout
                sys.stdout = output_buffer
                try:
                    winner, sorted_statements = hm.mediate(valid_opinions)
                    result_container[0] = winner
                    result_container[1] = sorted_statements
                except Exception as e:
                    exception_container[0] = e
                finally:
                    sys.stdout = old_stdout

            # Start deliberation in background thread
            thread = threading.Thread(target=run_mediate)
            thread.start()

            # Monitor progress and update status
            phase = "generating"
            while thread.is_alive():
                output = output_buffer.getvalue()

                # Check if we've moved to ranking phase
                if "Statements generated:" in output and phase == "generating":
                    phase = "ranking"
                    status.write("✅ Candidate statements generated")
                    status.write(f"🔄 Predicting rankings for {len(valid_opinions)} participants...")

                time.sleep(0.1)

            # Wait for thread to complete
            thread.join()

            # Check for exceptions
            if exception_container[0]:
                raise exception_container[0]

            # Get results
            winner, sorted_statements = result_container
            st.session_state.winner = winner
            st.session_state.sorted_statements = sorted_statements
            st.session_state.critiques = [""] * len(valid_opinions)

            status.write("✅ Rankings computed")
            status.update(label="✅ Deliberation complete!", state="complete")

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

    # Google Sheets import for critiques
    if SHEETS_AVAILABLE:
        with st.expander("📊 Import Critiques from Google Sheets (Optional)"):
            st.markdown("""
            **How to use:**
            1. Share the winning statement with students
            2. Create a new Google Form asking: "What would you change about this statement?"
            3. Make the response Sheet **publicly viewable**
            4. Paste the Sheet URL below and click Import

            **Expected format:**
            - Column A: Critique responses (one per row, starting from row 2)
            """)

            critique_sheet_url = st.text_input(
                "Google Sheets URL for Critiques",
                placeholder="https://docs.google.com/spreadsheets/d/...",
                help="The sheet must be publicly viewable",
                key="critique_sheet_url"
            )

            critique_col = st.text_input("Critique Column", value="B", max_chars=1, key="critique_col")

            if st.button("📥 Import Critiques from Sheet"):
                if not critique_sheet_url:
                    st.error("Please enter a Google Sheets URL")
                else:
                    try:
                        with st.spinner("Fetching critiques from Google Sheets..."):
                            _, imported_critiques = fetch_from_google_sheets(
                                critique_sheet_url,
                                opinion_column=critique_col,
                                question_column=None
                            )

                            if not imported_critiques:
                                st.error("No critiques found in the sheet. Check the column letter.")
                            else:
                                # Update session state with imported critiques
                                st.session_state.critiques = imported_critiques

                                # Also set individual widget states
                                for i, crit in enumerate(imported_critiques):
                                    st.session_state[f"critique_{i}"] = crit

                                st.success(f"✅ Imported {len(imported_critiques)} critiques from Google Sheets!")

                                # Force a rerun to show the imported data
                                st.rerun()

                    except Exception as e:
                        st.error(f"Error importing critiques: {str(e)}")
                        st.markdown("""
                        **Troubleshooting:**
                        - Make sure the Sheet is set to "Anyone with the link can view"
                        - Check that the column letter is correct
                        """)

    # Critique inputs. Number of critique boxes mirrors the number of
    # non-empty opinions; paginate the same way as the opinion section.
    num_critique_boxes = len(
        [op for op in st.session_state.opinions if op.strip()]
    )

    # Make sure the critiques list is long enough.
    if len(st.session_state.critiques) < num_critique_boxes:
        st.session_state.critiques.extend(
            [""] * (num_critique_boxes - len(st.session_state.critiques))
        )

    CRITIQUE_PAGE_SIZE = 25

    # Seed widget state so off-page critiques are preserved.
    for i in range(num_critique_boxes):
        if f"critique_{i}" not in st.session_state:
            st.session_state[f"critique_{i}"] = st.session_state.critiques[i]

    if num_critique_boxes > CRITIQUE_PAGE_SIZE:
        c_num_pages = (num_critique_boxes - 1) // CRITIQUE_PAGE_SIZE + 1
        c_page = st.number_input(
            f"Page (showing {CRITIQUE_PAGE_SIZE} critiques per page)",
            min_value=1,
            max_value=c_num_pages,
            value=1,
            key="critique_page",
        )
        c_start = (c_page - 1) * CRITIQUE_PAGE_SIZE
        c_end = min(c_start + CRITIQUE_PAGE_SIZE, num_critique_boxes)
        st.caption(
            f"Showing critiques {c_start + 1}–{c_end} "
            f"of {num_critique_boxes}"
        )
    else:
        c_start, c_end = 0, num_critique_boxes

    for i in range(c_start, c_end):
        st.text_area(
            f"Critique from Participant {i + 1}",
            height=80,
            key=f"critique_{i}",
        )

    # Sync widget state back into the critiques list for every participant.
    for i in range(num_critique_boxes):
        if f"critique_{i}" in st.session_state:
            st.session_state.critiques[i] = st.session_state[f"critique_{i}"]

    # Run critique round (gated behind a cost-estimate confirmation dialog).
    critique_btn_clicked = st.button(
        "🔄 Run Critique Round", type="secondary", use_container_width=True
    )

    if critique_btn_clicked:
        _valid_critiques = [
            c.strip() for c in st.session_state.critiques if c.strip()
        ]
        if len(_valid_critiques) < 2:
            st.error(
                "Please enter at least 2 critiques to run the critique round."
            )
        else:
            _valid_opinions_for_estimate = [
                op.strip() for op in st.session_state.opinions if op.strip()
            ]
            _estimate = estimate_cost(
                question=question,
                opinions=_valid_opinions_for_estimate,
                num_candidates=num_candidates,
                model_name=model_name,
                previous_winner=st.session_state.winner,
                critiques=_valid_critiques,
            )
            st.session_state.pending_critique_run = {
                'estimate': _estimate,
                'num_participants_in_run': len(_valid_critiques),
            }

    if st.session_state.pending_critique_run is not None:
        _pending = st.session_state.pending_critique_run
        _confirm_critique_dialog(
            _pending['estimate'], _pending['num_participants_in_run']
        )

    if st.session_state.confirmed_critique_run:
        st.session_state.confirmed_critique_run = None
        valid_critiques = [
            c.strip() for c in st.session_state.critiques if c.strip()
        ]

        print("\n" + "="*80)
        print(f"STARTING CRITIQUE ROUND")
        print(f"Number of critiques: {len(valid_critiques)}")
        print("="*80 + "\n")

        try:
            with st.status("Running critique round...", expanded=True) as status:
                status.write("🔄 Generating refined candidate statements...")

                # Capture stdout to monitor progress
                output_buffer = io.StringIO()
                result_container = [None, None]  # [winner, sorted_statements]
                exception_container = [None]

                # Pull hm into a local variable so the worker thread doesn't
                # touch st.session_state (which isn't safe without a
                # ScriptRunContext and fails outright on newer Streamlit).
                hm = st.session_state.hm

                def run_critique_mediate():
                    """Run mediation in thread with output capture."""
                    old_stdout = sys.stdout
                    sys.stdout = output_buffer
                    try:
                        winner, sorted_statements = hm.mediate(valid_critiques)
                        result_container[0] = winner
                        result_container[1] = sorted_statements
                    except Exception as e:
                        exception_container[0] = e
                    finally:
                        sys.stdout = old_stdout

                # Start critique round in background thread
                thread = threading.Thread(target=run_critique_mediate)
                thread.start()

                # Monitor progress and update status
                phase = "generating"
                while thread.is_alive():
                    output = output_buffer.getvalue()

                    # Check if we've moved to ranking phase
                    if "Statements generated:" in output and phase == "generating":
                        phase = "ranking"
                        status.write("✅ Refined statements generated")
                        status.write(f"🔄 Predicting rankings for {len(valid_critiques)} participants...")

                    time.sleep(0.1)

                # Wait for thread to complete
                thread.join()

                # Check for exceptions
                if exception_container[0]:
                    raise exception_container[0]

                # Get results
                winner, sorted_statements = result_container

                status.write("✅ Rankings computed")
                status.update(label="✅ Critique round complete!", state="complete")

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
