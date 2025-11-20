# Prompt- und Tool-Referenz

Zentrale Ablage für die Agent-Prompts und Tool-Spezifikationen. Dieser Überblick
fasst die aktuell verwendeten Prompts (siehe `src/graphrag/agent.py`) und die
wichtigen Guardrails zusammen.

## System-Prompt (Kernaussagen)
- Erzwingt Tool-Nutzung für jede Antwort und verbietet Halluzinationen.
- Empfiehlt einen Schema-Awareness-Schritt über das Tool `schema_overview`,
  bevor neue Cypher-Queries erzeugt werden.
- Beschreibt die vier verfügbaren Tools (`schema_overview`, `semantic_search`,
  `hybrid_retrieve`, `cypher_query`) inklusive Argumente und Beispielaufrufe.
- Enthält Multi-Hop-Heuristiken sowie klare Fehler- und Guardrail-Hinweise
  (z. B. [ERR_*]-Codes, keine Schreiboperationen, nur verfügbare Properties
  verwenden).

## Tool-Spezifikationen
- **schema_overview**: Liefert Labels, Relationship-Typen und bekannte
  Node-Properties, um Cypher-Queries schema-konform zu halten.
- **semantic_search**: Semantische/Vektor-Suche; `query`, optional `top_k`.
- **hybrid_retrieve**: Kombiniert Vektor-, Keyword- und Graph-Suche; optional
  `top_k` und `expand_hops`. Ergebnisse werden nach einer gewichteten Fusion
  und einem konfigurierbaren Reranker sortiert.
- **cypher_query**: Read-only-Cypher; Guardrail blockiert Schreibbefehle.
  Fehler werden mit `[ERR_CYPHER_*]` gekennzeichnet.

## Fehlercodes & Telemetrie
- Tool-Ausgaben nutzen strukturierte Fehlercodes (`[ERR_UNKNOWN_TOOL]`,
  `[ERR_CYPHER_GUARDRAIL]`, `[ERR_TOOL_EXCEPTION]` usw.), damit der Agent
  Fehlpfade erkennen und anpassen kann.
- Der `GraphRAGToolExecutor` protokolliert Latenzen pro Tool (p50/p95/max);
  abrufbar über `tool_executor.telemetry_summary()` und in der UI einsehbar.

## Conversational Memory
- Eine optionale `ConversationMemory`-Instanz puffert die letzten
  Chat-Nachrichten (mit Limit) und ermöglicht Folgefragen ohne Verlust des
  bisherigen Kontexts. Sie wird in der UI automatisch aktiviert.
