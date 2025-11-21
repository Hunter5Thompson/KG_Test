# Vorschläge zur Verbesserung des GraphRAG Agents

Diese Liste fasst kurzfristig umsetzbare Verbesserungen zusammen, nachdem der aktuelle Funktionsumfang (Agent, Tools, UI, Tests) überprüft wurde.

## Zuverlässigkeit & Guardrails
- Ergänze einen expliziten Schema-Awareness-Schritt vor Cypher-Generierung (Schema introspection + Kurzanleitung im Prompt), damit der Agent zielgerichtete Lese-Queries erzeugt und Fehlversuche reduziert.
- Ergänze strukturierte Fehlercodes für Tool-Ausgaben (z. B. `ERR_CYPHER_VALIDATION`, `ERR_EMBEDDING`) und parse sie in der Antwort-Synthese, um Fehlversuche im ReAct-Loop zu minimieren.

## Retrieval-Qualität
- Ergänze konfigurierbare Reranker (Cross-Encoder oder LLM-Score) nach `HybridGraphRetriever.retrieve`, um Kontextqualität zu steigern, bevor die Antwort erzeugt wird.
- Füge Telemetrie für Tool-Latenzen (Histogramme) hinzu, damit langsame Pfade und Engpässe schnell sichtbar werden.

## Gesprächskontext & Nutzbarkeit
- Baue einen optionalen Conversational Memory (z. B. Buffer mit Token-Limit) auf Basis von `AgentState.messages`, damit Mehrschritt-Dialoge nicht jedes Mal vom Scratch starten.
- Aktiviere Streaming-Antworten im UI (`src/ui/agent_ui.py`) und API, damit Nutzer Fortschritt sehen, während Tool-Calls laufen.

## Wartbarkeit & Tests
- Ergänze Unit- und Integrationstests für `GraphRAGToolExecutor` (inkl. Fehlerpfade wie ungültige Cypher-Queries), damit Regressionen früh erkannt werden.
- Dokumentiere die Prompt-Templates und Tool-Spezifikationen zentral (z. B. in `config/prompts/`), um Anpassungen versionssicher und nachvollziehbar zu machen.
