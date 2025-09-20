import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from river import compose, naive_bayes, stream
from river import feature_extraction as fx
from tabulate import tabulate

st.set_page_config(layout="wide", page_title="Online vs. Batch Learning")

# --- 1. Data Generation (Same as before, modified for streamlit) ---

@st.cache_data
def generate_data(ham_keywords, spam_keywords, num_samples):
    """Generates synthetic text data with varying distributions.
    
    Args:
        ham_keywords: List of ham keywords
        spam_keywords: List of spam keywords
        num_samples: Total number of samples to generate
        
    Returns:
        Tuple of (data, labels) where data contains messages and labels are 0/1 for ham/spam
    """
    data = []
    labels = []
    
    # Split into training phases
    phase1_end = num_samples // 3
    phase2_end = 2 * num_samples // 3
    
    for i in range(num_samples):
        if i < phase1_end:
            # Phase 1: 60% ham, 40% spam
            is_spam = np.random.random() < 0.4
        elif i < phase2_end:
            # Phase 2: 50% ham, 50% spam
            is_spam = np.random.random() < 0.5
        else:
            # Phase 3: 40% ham, 60% spam
            is_spam = np.random.random() < 0.6
            
        if not is_spam:
            # Generate ham message with some variation
            num_ham_words = np.random.choice([3, 4, 5])
            msg = " ".join(np.random.choice(ham_keywords, num_ham_words))
            data.append(msg)
            labels.append(0)
        else:
            # Generate spam message with some variation
            num_spam_words = np.random.choice([4, 5, 6])  # Spam messages might be slightly longer
            msg = " ".join(np.random.choice(spam_keywords, num_spam_words))
            data.append(msg)
            labels.append(1)
            
    return data, labels

def generate_hard_drift_data(ham_keywords, new_spam_keywords, num_samples):
    """Generates drifted spam with gradual concept drift.
    
    Args:
        ham_keywords: List of ham keywords
        new_spam_keywords: List of new spam keywords
        num_samples: Number of samples to generate
        
    Returns:
        Tuple of (data, labels) where data contains messages and labels are 1 for spam
    """
    data = []
    labels = []
    
    # Split the samples into phases
    phase1_end = num_samples // 3
    phase2_end = 2 * num_samples // 3
    
    for i in range(num_samples):
        if i < phase1_end:
            # Phase 1: Mix of old and new spam words (25% new)
            if np.random.random() < 0.25:
                msg = " ".join(np.random.choice(new_spam_keywords, 5))
            else:
                msg = " ".join(np.random.choice(ham_keywords, 5))
            # 80% chance of being spam in this phase
            is_spam = np.random.random() < 0.8
            
        elif i < phase2_end:
            # Phase 2: More new words (50% new)
            if np.random.random() < 0.5:
                msg = " ".join(np.random.choice(new_spam_keywords, 5))
            else:
                msg = " ".join(np.random.choice(ham_keywords, 5))
            # 90% chance of being spam in this phase
            is_spam = np.random.random() < 0.9
            
        else:
            # Phase 3: Mostly new words (75% new)
            if np.random.random() < 0.75:
                msg = " ".join(np.random.choice(new_spam_keywords, 5))
            else:
                msg = " ".join(np.random.choice(ham_keywords, 5))
            # 95% chance of being spam in this phase
            is_spam = np.random.random() < 0.95
        
        data.append(msg)
        labels.append(1 if is_spam else 0)
        
    return data, labels

# --- 2. Model Simulation and Evaluation ---

def run_simulation(progress_bar, status_text, results_placeholder):
    # Data generation
    initial_ham_keywords = ["meeting", "project", "report", "lunch", "team", "task"]
    initial_spam_keywords = ["free", "prize", "winner", "cash", "congratulations", "claim"]
    new_spam_keywords = ["crypto", "blockchain", "investment", "financial", "trading"]
    initial_X, initial_y = generate_data(initial_ham_keywords, initial_spam_keywords, num_samples=1000)
    drift_X, drift_y = generate_hard_drift_data(initial_ham_keywords, new_spam_keywords, num_samples=50)
    stream_X = initial_X[:200] + drift_X
    stream_y = initial_y[:200] + drift_y

    # Batch Model
    vectorizer = CountVectorizer()
    X_initial_vectorized = vectorizer.fit_transform(initial_X)
    batch_model = MultinomialNB()
    batch_model.fit(X_initial_vectorized, initial_y)

    # Online Model
    online_model = compose.Pipeline(
        ('vectorizer', fx.BagOfWords(tokenizer=str.split)),
        ('nb', naive_bayes.MultinomialNB())
    )
    for x, y in zip(initial_X, initial_y):
        online_model.learn_one(x, y)

    # Simulation tracking
    online_predictions = []
    batch_predictions = []
    drift_online_correct = []
    
    # Track confidence scores
    online_confidence = []
    batch_confidence = []
    
    # Track learning progress
    learning_progress = []
    
    total_drifted_spam = len(drift_X)
    print(f"Total drifted spam messages: {total_drifted_spam}")
    print(f"Drift X sample: {drift_X[:2] if len(drift_X) > 0 else 'Empty'}")
    print(f"Drift y sample: {drift_y[:2] if len(drift_y) > 0 else 'Empty'}")
    
    online_correct_after_learn_count = 0
    batch_missed_drifted_spam_count = 0
    online_total_drifted_spam = 0
    results_table_data = []
    
    # Track model's understanding of new patterns
    new_patterns_learned = 0
    model_updates = 0

    for i, (text, true_label) in enumerate(zip(stream_X, stream_y)):
        progress_percentage = int((i + 1) / len(stream_X) * 100)
        progress_bar.progress(progress_percentage)

        # Batch prediction with confidence
        batch_proba = batch_model.predict_proba(vectorizer.transform([text]))[0]
        batch_pred = 1 if batch_proba[1] > 0.5 else 0
        batch_conf = max(batch_proba)  # Confidence is the higher probability
        
        # Online prediction before learning
        online_pred_pre = online_model.predict_one(text)
        
        # Get pre-learning confidence (if possible)
        try:
            online_conf_pre = online_model.predict_proba_one(text).get(1, 0.5)
        except:
            online_conf_pre = 0.5  # Default confidence if not available
        
        # Online learning
        online_model.learn_one(text, true_label)
        model_updates += 1
        
        # Online prediction after learning
        online_pred_post = online_model.predict_one(text)
        
        # Get post-learning confidence
        try:
            online_conf_post = online_model.predict_proba_one(text).get(1, 0.5)
        except:
            online_conf_post = 0.5
            
        # Track if model learned something new
        if online_pred_pre != online_pred_post:
            new_patterns_learned += 1

        # Track performance and confidence
        online_correct = online_pred_post == true_label
        batch_correct = batch_pred == true_label
        
        online_predictions.append(online_correct)
        batch_predictions.append(batch_correct)
        online_confidence.append(online_conf_post)
        batch_confidence.append(batch_conf)
        
        is_drift_message = text in drift_X
        if is_drift_message:
            online_total_drifted_spam += 1
            drift_online_correct.append(online_correct)
            
            # Detailed logging for drifted messages
            print(f"\n--- Drifted Message {online_total_drifted_spam} ---")
            print(f"Text: {text}")
            print(f"True label: {'SPAM' if true_label else 'HAM'}")
            print(f"Batch: {'CORRECT' if batch_correct else 'WRONG '} (Confidence: {batch_conf:.2f})")
            print(f"Online pre-learn: {'SPAM' if online_pred_pre else 'HAM'} (Confidence: {online_conf_pre:.2f})")
            print(f"Online post-learn: {'SPAM' if online_pred_post else 'HAM'} (Confidence: {online_conf_post:.2f})")
            
            # Track learning progress
            learning_point = {
                'message_num': i,
                'text': text,
                'batch_correct': batch_correct,
                'online_pre_correct': online_pred_pre == true_label,
                'online_post_correct': online_correct,
                'batch_confidence': batch_conf,
                'online_confidence': online_conf_post,
                'learning_happened': online_pred_pre != online_pred_post
            }
            learning_progress.append(learning_point)
            
            # Print learning insights
            if online_pred_pre != online_pred_post:
                print("  -> Model updated its prediction after learning!")

        # Log to table
        if is_drift_message:
            if batch_pred != true_label:
                batch_missed_drifted_spam_count += 1
            if online_pred_post == true_label:
                online_correct_after_learn_count += 1
            results_table_data.append({
                "New Spam #": online_total_drifted_spam,
                "Batch Pred": "SPAM" if batch_pred else "HAM",
                "Online (pre-learn)": "SPAM" if online_pred_pre else "HAM",
                "Online (post-learn)": "SPAM" if online_pred_post else "HAM",
                "Outcome": "Online learns from error" if online_pred_pre != online_pred_post else "Online blocks repeat"
            })

        status_text.text(f"Processing message {i+1}/{len(stream_X)}...")

    # Post-simulation processing and display
    progress_bar.empty()
    status_text.empty()

    print("\n--- Simulation Complete ---")
    print(f"Total messages processed: {len(stream_X)}")
    print(f"Drifted spam messages found: {online_total_drifted_spam}")
    print(f"Results table data length: {len(results_table_data)}")

    results_placeholder.header("Simulation Results")
    results_placeholder.markdown("""
        The simulation is complete. Below are the performance comparisons.
        The batch model was trained once on the initial spam patterns and does not adapt.
        The online model was trained on the same data but continues to learn from the new stream.
    """)

    # Convert learning progress to DataFrame for visualization
    learning_df = pd.DataFrame(learning_progress) if learning_progress else pd.DataFrame()
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Performance Overview", "Learning Progress", "Detailed Analysis"])
    
    with tab1:
        st.subheader("Model Performance Summary")
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Batch Accuracy on Drift", 
                     f"{100 * (1 - batch_missed_drifted_spam_count/max(1, total_drifted_spam)):.1f}%",
                     f"Missed {batch_missed_drifted_spam_count} of {total_drifted_spam}")
        with col2:
            st.metric("Online Accuracy on Drift",
                     f"{100 * (online_correct_after_learn_count/max(1, total_drifted_spam)):.1f}%",
                     f"Correctly classified {online_correct_after_learn_count} of {total_drifted_spam}")
        with col3:
            st.metric("Patterns Learned", 
                     f"{new_patterns_learned}",
                     f"Model updated {model_updates} times")
        
        # Performance over time
        st.subheader("Accuracy Over Time")
        accuracy_data = pd.DataFrame({
            'Batch': pd.Series(batch_predictions).expanding().mean(),
            'Online': pd.Series(online_predictions).expanding().mean(),
            'Drifted Messages': [i in [lp['message_num'] for lp in learning_progress] for i in range(len(online_predictions))]
        })
        st.line_chart(accuracy_data[['Batch', 'Online']])
        
        # Confidence comparison
        st.subheader("Confidence Levels")
        if not learning_df.empty:
            conf_data = learning_df[['batch_confidence', 'online_confidence']]
            conf_data.index = learning_df['message_num']
            st.line_chart(conf_data.rename(columns={
                'batch_confidence': 'Batch Confidence',
                'online_confidence': 'Online Confidence'
            }))
    
    with tab2:
        st.subheader("Learning Progress")
        if not learning_df.empty:
            # Learning events
            learning_events = learning_df[learning_df['learning_happened']]
            if not learning_events.empty:
                st.write(f"### Key Learning Events ({len(learning_events)} total)")
                for _, event in learning_events.iterrows():
                    with st.expander(f"Message {event['message_num']}: {event['text'][:50]}..."):
                        st.write(f"**Text:** {event['text']}")
                        st.write(f"**Batch:** {'Correct' if event['batch_correct'] else 'Incorrect'}")
                        st.write(f"**Online Before:** {'Correct' if event['online_pre_correct'] else 'Incorrect'}")
                        st.write(f"**Online After:** {'Correct' if event['online_post_correct'] else 'Incorrect'}")
                        st.write(f"**Confidence Change:** {event['online_confidence'] - online_conf_pre:.2f}")
            else:
                st.info("No significant learning events detected in this simulation.")
            
            # Learning rate over time
            st.write("### Learning Rate Over Time")
            if len(learning_events) > 1:
                learning_rate = learning_events['message_num'].value_counts().sort_index().cumsum()
                st.line_chart(learning_rate)
        else:
            st.info("No learning progress data available.")
    
    with tab3:
        st.subheader("Detailed Analysis")
        if not learning_df.empty:
            # Show raw data
            st.write("### Learning Events Data")
            st.dataframe(learning_df.drop('text', axis=1))
            
            # Show accuracy by phase
            if 'phase' in learning_df.columns:
                st.write("### Performance by Phase")
                phase_perf = learning_df.groupby('phase')[['batch_correct', 'online_post_correct']].mean()
                st.bar_chart(phase_perf * 100)
        else:
            st.info("No detailed analysis data available.")
    
    # Final summary
    st.success("""
    ### Key Insights:
    - The online model adapts to new patterns in real-time
    - Confidence levels show how certain the model is in its predictions
    - Learning events highlight when the model updates its understanding
    - Performance metrics demonstrate the advantage of continuous learning
    """)


# --- Streamlit App Layout ---

st.title("Online vs. Batch Learning with Concept Drift")
st.markdown("""
This application demonstrates how **online learning** models can adapt to **concept drift**, a phenomenon where the patterns underlying data change over time.

In this demo, we simulate a spam detector where:
*   The **batch model** is trained once on an initial set of spam keywords and never updated.
*   The **online model** is trained on the same data but keeps learning from new messages.

We then inject a stream of "hard" drifted spam containing new keywords.
""")

if st.button("Start Simulation"):
    with st.spinner("Running simulation..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_placeholder = st.empty()
        run_simulation(progress_bar, status_text, results_placeholder)