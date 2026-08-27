---
name: agentic-ecology-ui
description: >-
  Guidelines and templates for creating a generic web UI to visualize, rank,
  filter, and annotate vector databases. The UI has a hierarchical structure
  covering database selection, label selection/definition, ranking/filtering
  configuration, and row visualization/annotation. Meant to be copied and
  adapted for specific downstream tasks like bioacoustics or camera traps.
---

# UI Skill: Visualizing, Ranking, Filtering, and Annotating Vector Databases

This skill provides guidelines and templates for building a lightweight web UI
to visualize, rank, filter, and annotate vector databases. The databases contain
vector embeddings computed from raw inputs (e.g., audio segments, camera trap
images, or text segments), where each row corresponds to one embedding.

> [!CAUTION] **DATABASE SAFETY AND INTEGRITY**: Do NOT write, modify, or insert
> test annotations directly into the user's production databases. If you need to
> test database operations (such as saving annotations or training models), you
> MUST copy the database to a temporary location (e.g., inside the conversation
> scratch directory) and test against the copy. Never leave testing data in
> production databases.

The skill's output is generic, implementing common features and leaving specific
implementation details (like media rendering or custom inference models) to
downstream skills (e.g., bioacoustics, camera traps).

## UI Architecture

The interface follows a hierarchical four-level design:

1.  **Database Selection (Level 1):** Allows the user to select which database
    to browse from a list of databases present in a specified directory.
2.  **Label Selection/Definition (Level 2):** Allows the user to select an
    annotation label to focus on (with options determined dynamically based on
    annotations present in the database), or to define a new annotation label.
3.  **Ranking and Filtering Configuration (Level 3):** Allows the user to
    determine how to rank and filter rows in the database (e.g., by similarity
    with a query vector, classifier score, sorting order, paging, or filtering
    out already labeled items).
4.  **Row Visualization and Annotation (Level 4):** Allows the user to visualize
    the selected database's rows, along with their annotations for the active
    label, ranked and filtered according to the Level 3 configuration.

## Architectural Patterns

When creating a web interface for agent results, follow these patterns:

### 1. API-Driven Design (Separation of Concerns)

-   **Static Frontend:** The HTML page (`index.html`) should be **100% static**
    (no hardcoded results). It should use standard JavaScript (`fetch()`) to
    request data from the backend and render it dynamically in the DOM.
-   **API Backend:** The Python server exposes a `/api/...` endpoint returning
    JSON data.
-   **Benefit:** Cosmetic changes (CSS, layout, pagination) only require editing
    the HTML/JS file and refreshing the browser.

### 2. In-Memory Caching

-   AI/ML model operations (like generating embeddings or running similarity
    searches) are expensive and slow (often multiple seconds).
-   **Pattern:** The backend server should cache search/inference on the first
    request. Subsequent API requests for the data should be served instantly
    (<5ms) from this cache.

### 3. Dynamic Media Streaming (On-the-Fly Slicing)

-   When displaying segments of large media files (e.g., 5-second windows from
    20MB audio files), **do not** pre-slice them to disk or embed them as
    base64.
-   **Pattern:** Create a streaming endpoint (e.g.,
    `/stream?file=...&start=...&end=...`). The server uses efficient libraries
    (like `soundfile` for audio) to **seek directly** to the start time and read
    only the requested range in-memory, streaming the bytes back instantly.
-   **Network Efficiency:** This is crucial when serving from a remote machine
    to a local browser via SSH port forwarding, as it drastically reduces
    network transfer.

### 4. Client-Side Dynamic Paging

-   Handle pagination in the browser using JavaScript.
-   **Pattern:** Load all results via the API, but hide/show rows dynamically.
-   **Smart Player Control:** When navigating pages, ensure any active media
    players are automatically paused so they don't continue playing in the
    background.

### 5. Robust Media Layouts (Preventing Squishing)

-   Avoid default compression of media players (like `<audio>` or `<video>`)
    inside flexible layouts (like Bootstrap tables or flexbox containers).
-   **Pattern:** Always define a clear, functional width for media players
    (e.g., `width: 250px;` or `width: 300px;`) and ensure their container or
    table column has sufficient space and prevents wrapping (e.g., `white-space:
    nowrap;`).
-   **Benefit:** Ensures play controls and progress bars remain fully visible
    and usable across different screen sizes.

### 6. Thread-Safe Multi-Threaded Backend (SQLite Concurrency)

-   Single-threaded backends block the entire event loop when transferring large
    files or streaming audio, causing simultaneous API requests to time out.
-   **Pattern:** Use a multi-threaded server
    (`socketserver.ThreadingTCPServer`).
-   **Database Isolation**: SQLite connections are thread-bound. Do not share a
    single connection across threads (which causes `ProgrammingError`). Instead,
    initialize and pool database connections in **thread-local storage** (e.g.,
    `threading.local()`).

### 7. Robust Media Streaming Connection Management

-   When users pause or switch tabs, the browser aborts the stream connection
    abruptly. This throws a `BrokenPipeError` or `ConnectionResetError` during
    server `.write()` calls.
-   **Pattern:** Always wrap audio streaming write calls in a `try-except` block
    to catch and silence these disconnects gracefully, preventing traceback
    bloat.

### 8. Zero-Latency Browser-Side Operations (Instant Sorting & Paging)

-   Making API requests for simple paging, sorting, or checkbox filtering (like
    hiding annotated rows) introduces unnecessary network lag.
-   **Pattern:** Fetch the complete database results once from the backend.
    Implement sorting, filtering (like toggling between showing/hiding labeled
    data), and paging **entirely in client-side JavaScript** (e.g. using
    standard `.sort()` and `.filter()` arrays).
-   **Smart Media Management**: When users change pages, ensure JavaScript loops
    through all active media players on the page and calls `.pause()` to prevent
    audio overlaps.

### 9. Thread-Safe Shared State Locking

-   In multi-threaded backends, different worker threads handle `/api/train`,
    `/api/search`, and `/api/annotation` in parallel. Modifying shared variables
    (like in-memory search result caches) without protection leads to race
    conditions.
-   **Pattern:** Always protect writes to shared global caches or model states
    using a mutual exclusion lock (`threading.Lock()`).

### 10. Safe JSON Serialization (Handling NumPy Types, NaN, and Infinity)

-   **NumPy Types**: AI/ML libraries (like USearch or NumPy itself) often return
    values as NumPy-specific types (e.g., `numpy.uint64` for IDs,
    `numpy.float32` for scores). Python's default `json.dumps()` does not
    support these and will raise a `TypeError: Object of type ... is not JSON
    serializable`.
    *   **Pattern**: Explicitly cast NumPy values (e.g. `int()`, `float()`), or
        use a custom `JSONEncoder` (like the `SafeEncoder` class defined in the
        server template) to automatically convert NumPy types to native Python
        types during serialization.
-   **NaN and Infinity**: AI/ML models often generate score/distance metrics
    containing `NaN` or `Infinity`. Python's default `json.dumps()` allows these
    values, but they result in invalid JSON that crashes JavaScript parser
    engines (`JSON.parse`) in the browser.
    *   **Pattern**: Sanitize all float values using the `sanitize_float` helper
        function in [server.py](references/server.py) before returning them in
        JSON API responses.

### 11. Material Design 3 (M3) Visual Conformance

-   The UI template follows the **Material Design 3 (M3)** design system
    ([m3.material.io](https://m3.material.io/)) with a light color scheme.
-   **Pattern — Token-Driven CSS**: All styling is driven by CSS custom
    properties (design tokens) defined in `:root`. The tokens cover:
    -   **Color**: Full M3 light scheme with primary, secondary, tertiary,
        error, surface, and outline role tokens. The template uses a teal seed
        color (`#006B5E`) appropriate for ecology, but can be re-seeded by
        swapping the `:root` color block.
    -   **Typography**: M3 type scale (Display, Headline, Title, Body, Label ×
        Large/Medium/Small) using Roboto.
    -   **Shape**: M3 shape scale (None → Extra-small → Small → Medium → Large →
        Extra-large → Full).
    -   **Elevation**: 6-level shadow system (Level 0–5).
    -   **Motion**: Duration and easing tokens for transitions.
    -   **State Layers**: Hover (8%), focus (10%), pressed (10%) opacity
        overlays.
-   **Component mapping**: Top App Bar (Small), Navigation Drawer (Standard),
    Segmented Buttons, Filled/Tonal/Outlined Buttons, Outlined Text Fields,
    Outlined Cards, and M3-styled snackbar notifications.
-   **No external CSS framework**: The template uses only pure CSS with custom
    properties — no Tailwind, Bootstrap, or other framework — to ensure
    authentic M3 token mapping and smaller payload.

### 12. Iframe-Safe Error Handling (No alert/confirm/prompt)

-   When the UI is served inside an iframe (e.g., as an Antigravity sidecar),
    browsers block `alert()`, `confirm()`, and `prompt()` in cross-origin iframe
    contexts. The dialog never shows, JavaScript execution suspends permanently,
    and the calling `async` function's `finally` block never runs — causing the
    UI to freeze.
-   **Pattern — Inline Snackbar Notifications**: Replace all `alert()` calls
    with a dynamically-created snackbar element that appears at the bottom of
    the viewport and auto-dismisses after 6 seconds. Use M3 inverse-surface
    colors for info messages and error-container colors for errors.
-   **Fetch Timeout**: Long-running server operations (like model training) must
    use an `AbortController` with a timeout to prevent indefinite hangs if the
    server becomes unresponsive.
-   **Error Message Extraction**: When the server returns an error response,
    always attempt to parse the JSON body and extract the `.error` field to
    display the actual server message instead of a generic failure string.

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

## Templates and Examples

The skill contains generic templates meant to be copied to the project's
working directory (e.g., `agent_workspace/`) and adapted:

*   **Static HTML Template**: [index.html](references/index.html) — M3-compliant
    hierarchical client-side UI with design tokens, database/label selection,
    paging, annotation, and inline snackbar notifications.
*   **Custom Python Server**: [server.py](references/server.py) — API server
    exposing endpoints to list databases, retrieve rows, train classifiers, and
    save annotations.
