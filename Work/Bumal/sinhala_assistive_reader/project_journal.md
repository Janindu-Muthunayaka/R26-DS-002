# Research Project Journal: Sinhala Assistive Reader

## 📖 Project Overview
This project is developing an "**Adaptive Conversational Personalization and Voice Interaction Module**" (Component 4 of a larger system) for a Sinhala Assistive Reader. The module processes voice commands (STT), detects user intents, predicts preferred communication styles using an online learning model, and passes personalized prompt modifiers to a downstream RAG generation module (Component 3). 

The core research contribution lies in the **Adaptive Personalization Stage**, which employs online learning with no upfront training dataset, adapting dynamically to user feedback and implicit corrections in real-time.

---

## 🟢 Phase 1: Intent Detection (Completed)
The first phase focused on translating Sinhala voice inputs and accurately detecting the user's intent. Three approaches were considered and evaluated:

### Approaches Evaluated
1.  **Approach 1 (NLLB + Llama 3.2:1b)**: Uses NLLB for Sinhala-to-English translation, followed by an LLM (Llama 3.2:1b via Ollama) to extract intents via prompting. 
    *   *Status*: **Selected** as the primary intent detection method for the personalization flow.
2.  **Approach 2 (Direct LLM)**: Direct processing (not used going forward).
3.  **Approach 3 (Trained TF-IDF + LinearSVC)**: A fast, lightweight trained classifier distilling Approach 1's behavior. We conducted a large-scale evaluation of this approach.
    *   *Evaluation Details*: Evaluated on 500 generated samples (`voice_interaction/data/generate_samples.py`). Built a custom batched PyTorch execution script to handle NLLB translations efficiently.
    *   *Results*: Achieved **91.40% accuracy** (457/500 correct) with an average classification time of ~2.05ms per sample.
    *   *Insights*: The classifier occasionally struggled with semantic overlap (e.g., confusing `SIMPLIFY` with `EXPLAIN`), mid-sentence self-corrections (e.g., "next... no, previous"), and nuanced navigation (page vs. article).

---

## 🟡 Phase 2: Adaptive Personalization (Current Phase & Active Blueprint)
We are currently implementing the **Personalization Module** according to `personalization_build_spec_v2.md`. 
The module logs interactions, detects repeat failures, predicts user communication styles (`Simple`, `Detailed`, `StepByStep`), and corrects its own model based on user feedback.

### Architecture & Tech Stack
*   **Database**: TinyDB (`db.json`) for lightweight local storage of user profiles and interaction logs.
*   **Machine Learning**: `river` library for online learning (TFIDF + SoftmaxRegression).
*   **Persistence**: The model saves state to disk (`style_model.pkl`) to retain learning across application restarts.

### Implementation Blueprint (Step-by-Step Plan)
The implementation is broken down into four distinct steps. This is the exact blueprint being executed:

#### Step 1: Logging System (`personalization/logger.py`)
*   **Goal**: Log every interaction into TinyDB and provide functions to retrieve and update specific records.
*   **Tasks**:
    *   Create `log_interaction` to append `detect_intent_approach1` results to the `interaction_logs` table.
    *   Create `get_last_interaction` to fetch a user's most recent interaction.
    *   Create `update_last_interaction_style` and `update_interaction_style_by_timestamp` to support relabeling past interactions when a user issues a correction on a subsequent turn.

#### Step 2: Diagnostic Checks (`personalization/diagnostic.py`)
*   **Goal**: Pure logic functions (no ML) to detect repeat failures and correction signals.
*   **Tasks**:
    *   Implement `is_repeat_failure()`: Checks if the user is asking to `REPEAT` the exact same content chunk immediately after the previous turn (bypasses ML and triggers a TTS replay).
    *   Implement `detect_correction_signal()`: Checks if the current intent (e.g., `SIMPLIFY` or `ELABORATE`) implies the *previous* turn's style guess was incorrect. Returns the corrected style if applicable.

#### Step 3: Online Learning Model (`personalization/style_model.py`)
*   **Goal**: Implement the River online learning pipeline with automatic disk persistence.
*   **Tasks**:
    *   Create a `compose.Pipeline` with `TFIDF()` and `SoftmaxRegression()`.
    *   Implement `_load_state()` and `_save_state()` using `pickle` to persist the model to `data/style_model.pkl`.
    *   Implement `predict_style()`: Predicts `Simple`, `Detailed`, or `StepByStep` (defaults to `Detailed` if untrained).
    *   Implement `learn_style()`: Updates the model with the true/corrected label and immediately saves the model to disk to prevent data loss.

#### Step 4: Main Flow Integration (`personalization/main_flow.py`)
*   **Goal**: Wire Steps 1-3 together into the full end-to-end pipeline.
*   **Tasks**: Implement `handle_voice_command()` which executes the following sequence:
    1.  Run intent detection (via Approach 1).
    2.  Fetch the previous interaction.
    3.  **Correction Loop**: Check if the current intent corrects the previous turn (Step 2). If so, update the database (Step 1) and re-learn the previous text with the corrected label (Step 3).
    4.  **Repeat Failure Check**: If it's a repeat request for the same chunk, route to `TTS_REPLAY`.
    5.  Log the current interaction.
    6.  Predict the style for the current turn and generate the `prompt_modifier`.
    7.  Update the log with the predicted style and run a standard learning update for the current turn.
    8.  Return the final payload for the downstream RAG generation module.

---

## 🔵 Future Work & Known Limitations
*   `retrieved_chunk_id` is currently manually passed as a placeholder string. It will be wired up once the RAG module (Component 3) is fully integrated.
*   Correction detection currently relies on explicit intents (`SIMPLIFY`, `ELABORATE`). Catching implicit corrections (e.g., "that was too long") requires broader intent coverage in the future.
*   TF-IDF on short voice commands relies on literal word overlap. Future improvements may include character n-grams or synonym mapping to improve generalization across synonyms (e.g., "shorter" vs. "brief").
