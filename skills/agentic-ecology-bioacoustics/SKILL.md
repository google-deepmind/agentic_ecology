---
name: agentic-ecology-bioacoustics
description: >-
  Provides bioacoustic analysis capabilities for ecologists and researchers
  using the perch-hoplite Python package. A typical use case is to use agile
  modeling to bootstrap the creation and deployment of a bespoke detector for
  targeted species on an existing collection of passive acoustic monitoring
  recordings.
---

# Bioacoustics Skill

> [!CAUTION] **DATABASE SAFETY AND INTEGRITY**: Do NOT write, modify, or insert
> test annotations directly into the user's production databases. If you need to
> test database operations (such as saving annotations or training models), you
> MUST copy the database to a temporary location (e.g., inside the conversation
> scratch directory) and test against the copy. Never leave testing data in
> production databases.

## Workflow Overview

Follow these sequential steps:

1.  **Preprocess:** Create and populate a Hoplite database from the user's
    recordings.
    1.  **Identify Recordings:** Locate the user's recordings. If the location
        is not provided, ask the user.
    2.  **Select Embedding Model:** Confirm which embedding model to use (e.g.,
        `perch_v2`, `surfperch`). Query the user if they have not specified one.
    3.  **Create Database:** Initialize a new Hoplite database in the
        `databases` directory located in the repo's root directory, configured
        with the selected model's embedding dimension.
    4.  **Populate Database:** Extract embeddings from the recordings and
        populate the database with them along with metadata.
2.  **Build a Bioacoustics Web App:** Create an interactive webpage for the user
    to browse, search, and annotate audio snippets associated with the Hoplite
    database created in the previous step. Make sure the web app supports the
    following:
    *   **Browsing:** Design the UI so that the user can inspect rows in the
        database and listen to their associated audio.
    *   **Annotating:** Empower the user to attach annotations to rows in the
        database. Save, update, and clear user annotations (positive, negative,
        or uncertain) directly in the database under the "user" provenance tag
        as they interact.
    *   **Searching:** Empower the user to reorder rows in the database
        according to various criteria:
        *   **Vector Search:** Allow the user to present a search query in the
            form of a URI pointing to an audio clip. Embed it with the selected
            model and perform a search operation in the database. Use the result
            to rerank all rows in the database. Ensure that the query URI is
            only used for ranking, and any annotations submitted are saved under
            the active label (e.g., species name), NOT under the query URI
            itself.
        *   **Trained Classifier**: Once enough annotations are provided for a
            particular label (at least two positives and one negative, or two
            negatives and one positive), allow the user to search with a
            classifier trained on those annotations. Train a linear classifier
            using `perch-hoplite` APIs, and use the classifier's weights to
            score database rows and rerank them.

## Technical Reference

For detailed API usage, implementation instructions, and code examples, see the
`references/technical_reference.md`.

> [!IMPORTANT] To comply with repository environment rules, any references or
> template code you plan to use, test, or modify MUST first be copied into the
> `agent_workspace/` directory. Do not run or import them directly from
> `skills/...` to ensure all executions remain inside the workspace.

This reference covers:

*   Hoplite Database initialization and loading
*   Populating database with embeddings using `EmbedWorker`
*   Resolving physical audio files from database records
*   Agile Modeling setup and search implementation
*   Serving search results via the interactive UI
*   Processing user annotations (saving, clearing, and restoring state)
