# GradientHarbor — NVIDIA Nemotron + LangChain + Cryptographic AI Security

A local demo/hackathon application for CSV analytics and evidence-grounded answers. It uses **LangChain `ChatOpenAI` against NVIDIA's OpenAI-compatible API endpoint** for actual Nemotron calls. The NVIDIA API key is not included; no fake Nemotron responses are generated in real mode.

## Included capabilities

- **Nemotron multi-agent workflow via LangChain:** planner, SQL analyst, answer synthesis, and independent model verifier.
- **SQL defense in depth:** SQLGlot read-only policy, table allowlist, SQLite authorizer, and query-work limits.
- **Evidence retrieval:** local TF-IDF retrieval over Markdown/text knowledge-base documents.
- **Evidence integrity:** SHA-256 digest plus Ed25519 digital signature stored in a local manifest.
- **Tamper-evident audit:** JSONL hash chain, HMAC-SHA256 authentication, and Ed25519 signature per record.
- **Encryption at rest:** optional Fernet encryption for evidence documents added in the Evidence Vault.
- **Prompt-injection/jailbreak screening:** deterministic heuristic checks on user requests and retrieved passages; suspicious passages are excluded from model context.
- **Security Lab:** SQL policy testing, prompt-screen tests, key status, evidence signature verification, and audit verification.
- **Multimodal Studio:** encrypted image/audio/video storage, local SmolVLM image understanding, NVIDIA-hosted audio understanding (if the selected endpoint supports audio inputs), local Whisper transcription, optional video-frame extraction, and optional local diffusion image generation.

## Setup on Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set your own NVIDIA API key:

```dotenv
NVIDIA_API_KEY=your_real_key_here
NVIDIA_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
```

Get a key through NVIDIA's API Catalog / Build portal. Model availability and quotas can change. The default model and endpoint follow NVIDIA's OpenAI-compatible API documentation. Never commit `.env`.

Generate local cryptographic keys (optional; first app launch also creates them):

```powershell
python scripts/generate_keys.py
```

Run the application:

```powershell
streamlit run app.py
```

Run tests:

```powershell
python -m pytest -q
```

## Architecture

1. The prompt gate screens the request for known override, system-prompt extraction, jailbreak-persona, credential-exfiltration, and policy-bypass patterns.
2. LangChain calls NVIDIA Nemotron for the plan and a proposed SQL query.
3. SQLGlot validates that only a single allowed read-only query is used; SQLite authorization and work limits provide an independent execution boundary.
4. Retrieved evidence is scored locally. Suspicious prompt-injection passages and passages without a valid Ed25519 signature are excluded from model context. All evidence is still considered untrusted.
5. Nemotron synthesizes the answer from query results and remaining evidence, then a separate Nemotron call reviews it.
6. Audit events are hash chained and protected with HMAC-SHA256 and Ed25519 signatures.
7. Evidence can be stored encrypted at rest using Fernet. Existing/default knowledge-base files remain plaintext until re-added with the encryption option.

## Multimodal Studio (optional features)

The core app stays lightweight. Install only the optional model stack you want to try; these packages download model weights on first use.

### Local VLM (image understanding)

```powershell
pip install -r requirements-vlm.txt
```

Default model: `HuggingFaceTB/SmolVLM-500M-Instruct`. It runs locally on CPU, but inference may be slow and model downloads require internet access. The Hugging Face Transformers image-text-to-text pipeline is used for image + question input.

### ALM / audio understanding

The app offers two options:

- **NVIDIA-hosted audio-language analysis:** set `NVIDIA_API_KEY` and `NVIDIA_AUDIO_MODEL` in `.env`, then use **Understand audio with NVIDIA-hosted ALM**. The selected model and endpoint must support audio input via `audio_url`; availability varies by endpoint/model. The recording is sent to the hosted provider for inference. Do not send sensitive recordings unless your policy permits it. NVIDIA's multimodal API documentation describes audio input using `audio_url` content.
- **Local Whisper transcription:** install `requirements-alm.txt`; it uses `faster-whisper`'s `tiny` model by default for CPU use. This is speech-to-text, not full audio reasoning.

```powershell
pip install -r requirements-alm.txt
```

### Video frames (optional)

```powershell
pip install -r requirements-video.txt
```

The demo extracts at most three evenly-spaced frames from a video. If enabled, SmolVLM describes each selected frame. This is not continuous video understanding or forensic-grade analysis. OpenCV codec support varies; MP4/H.264 is the best-supported demo format.

### Diffusion image generation (optional)

```powershell
pip install -r requirements-diffusion.txt
```

Default model: `segmind/tiny-sd`, invoked through Hugging Face Diffusers. The app stores generated PNGs encrypted in the Media Vault. Model weights are several hundred MB or more depending on dependencies, and CPU generation can be very slow. Your 8 GB RAM laptop may struggle if VLM, diffusion and other applications are loaded simultaneously. Try one model at a time; a cloud GPU is preferable for diffusion. Check the model card/license before using generated content commercially.

### Encrypted Media Vault

Uploaded originals and generated images are encrypted into `media_store/*.enc`. `media_store/manifest.jsonl` contains metadata, the original SHA-256 digest, and an Ed25519 signature. The UI can decrypt and verify each file against its digest. Local development storage is not a substitute for production object storage or a KMS. Back up `.secrets/` securely; losing the encryption key makes stored media unrecoverable. Media uploads are limited to 100 MB by the application.

### Suggested install order for a laptop

Do not install every optional stack at once. Keep the core app working, then try one at a time:

1. `pip install -r requirements-vlm.txt` for image understanding.
2. `pip install -r requirements-alm.txt` for local speech transcription.
3. `pip install -r requirements-video.txt` for video frame extraction.
4. `pip install -r requirements-diffusion.txt` only if you have sufficient RAM/storage or a cloud GPU.

The optional stacks may require a compatible Python/PyTorch build; if installation fails, follow the official PyTorch install selector and the model's own requirements.

## Security notes — read before presenting this as secure

- The heuristic prompt-injection detector is not a state-of-the-art classifier and is not guaranteed to detect novel attacks. A clean scan does not prove safety. The UI and model prompts continue to treat external content as untrusted.
- Local keys are generated under `.secrets/` and are excluded by `.gitignore`. Anyone who obtains the Ed25519 private key can forge signatures; anyone who obtains the Fernet key can decrypt stored documents; anyone who obtains the HMAC key can forge authenticated audit records. Store keys outside the app host for production, ideally in a KMS/HSM or secrets manager.
- A local administrator can replace the application, delete logs, or steal local keys. Export signed audit checkpoints to an independently controlled system for meaningful tamper evidence.
- Losing the Fernet key means encrypted evidence cannot be recovered. Back up keys securely before encrypting important files.
- Ed25519 signatures authenticate data relative to the trusted public key; they do not prove the document's claims are true.
- The verifier is another LLM call and can be wrong. Human review is needed for high-impact decisions.
- NVIDIA API calls may be subject to quotas, latency, endpoint availability, and usage terms. The app has a separate local demo mode that explicitly does not call Nemotron.
- The sample retail dataset is synthetic. Do not upload secrets or personal data to an evaluation endpoint.

## Project structure

```text
app.py
app/main.py
gradientharbor/
  agents/workflow.py
  audit/hash_chain.py
  multimodal.py              # encrypted media vault + optional VLM/ALM/diffusion adapters
  provider/nvidia_client.py
  rag/retriever.py
  security/crypto.py
  security/integrity.py
  security/prompt_guard.py
  security/sql_guard.py
media_store/                 # runtime encrypted media files + signed manifest
knowledge_base/
data/retail_transactions.csv
requirements-vlm.txt
requirements-alm.txt
requirements-video.txt
requirements-diffusion.txt
tests/
```

## Official references

- Hugging Face Transformers image-text-to-text: https://huggingface.co/docs/transformers/tasks/image_text_to_text
- Hugging Face Diffusers pipelines: https://huggingface.co/docs/diffusers/en/api/pipelines/overview
- NVIDIA multimodal audio input: https://docs.nvidia.com/nim/large-language-models/latest/advanced-use-cases/multimodal-input.html

- NVIDIA Nemotron models and API: https://developer.nvidia.com/topics/ai/nemotron
- NVIDIA OpenAI-compatible LLM API: https://docs.api.nvidia.com/nim/reference/llm-apis
- LangChain NVIDIA integrations: https://github.com/langchain-ai/langchain-nvidia
- LangChain OpenAI-compatible chat integration: https://python.langchain.com/docs/integrations/chat/openai/

## Dashboard UI/UX and Plotly KPIs

The Overview tab includes dashboard-only city/category/date filters, KPI cards, Plotly trends and category/city/profitability charts, a data explorer, and CSV export.


## AI safety guardrails and agent observability

The Security Lab now includes a deterministic guardrail smoke-test suite and a dedicated **AI Safety & Agent Metrics** tab.

### Additional defenses

- Unicode NFKC normalization and removal of zero-width formatting characters before prompt screening.
- Request size limit (12,000 characters) and bounded untrusted-content scan.
- Pattern screening for instruction overrides, jailbreak personas, system-prompt extraction, credential exfiltration, tool-policy bypass, role spoofing, indirect instructions, prompt-boundary tokens, and suspicious base64-encoded instruction patterns.
- Risk score and severity (`low`, `medium`, `high`); high-risk requests are blocked and medium-risk requests are flagged for review.
- Retrieved evidence is screened and untrusted instructions are explicitly demoted in agent prompts.
- Model output is checked for credential-shaped strings and withheld if a configured secret pattern matches.
- Read-only SQL policy remains enforced before SQLite execution; the SQLite authorizer and query instruction limit remain in place.
- Agent and workflow telemetry records stage latency, success/error state, verifier status, guardrail decisions, and evidence counts without storing raw prompts, model answers, media bytes, or API keys in the metrics file.

### Agent metrics

The `AI Safety & Agent Metrics` tab shows workflow count/success rate, mean and p95 latency, per-stage median latency, guardrail blocks/review flags, and verifier outcomes. Telemetry is written to `audit_logs/agent_metrics.jsonl` at runtime. The Security Lab's fixed labeled examples report precision, recall, F1, and confusion counts for that tiny smoke-test set only.

### Important limitations

These are defense-in-depth controls, not a complete jailbreak detector or security certification. Heuristics can miss attacks and can flag benign content. A verifier model can be wrong. The local metrics and audit files can be modified by an attacker with filesystem access. Production deployments should add independent identity and authorization, rate limits, centralized immutable logging, KMS/HSM-backed key storage, sandboxed workers, tenant isolation, network egress controls, human approval for consequential actions, and a representative adversarial evaluation set. The current app does not autonomously execute external actions; keep it that way unless a separate tool authorization layer is added and tested.

## Read-only Vulnerability Investigation (CSV / Excel)

The **Vulnerability Investigation** tab scans CSV, XLSX, and XLSM uploads and produces a downloadable JSON/CSV report. It does not alter the uploaded file, connect to a live database, execute spreadsheet formulas, or automatically remediate findings.

### Categories covered

- **Data quality and poisoning candidates:** duplicate rows, missing-value concentrations, constant columns, and extreme numeric outlier candidates. These are signals for review, not proof of malicious poisoning.
- **Database/data exposure indicators visible in files:** potentially sensitive column names, common credential-shaped strings, email/card-like values, formula-like cells, hidden worksheets, external workbook links, and formula presence. This does **not** inspect live database grants, network exposure, cloud IAM, or server configuration.
- **AI-agent and prompt-injection candidates:** applies the project's heuristic prompt guard to text cells and reports signal names/counts without echoing the original cell contents.
- **Cryptographic integrity:** calculates a SHA-256 fingerprint and supports signed baseline manifests using the project's Ed25519 key. A baseline comparison can confirm a byte mismatch relative to that signed baseline. A signature created after upload does not prove the file was authentic before signing.

### Nemotron triage

The optional **Ask Nemotron to prioritize findings** action sends only the scanner's finding metadata—not raw CSV/Excel rows or cell values—to the configured NVIDIA endpoint through the LangChain client. Do not include sensitive values in column names or filenames. Model-generated prioritization is advisory and should be reviewed by a human.

### Integrity baseline workflow

1. Start with a known-good file whose provenance you have independently verified.
2. Upload it and choose **Create signed integrity baseline**.
3. Download and store the JSON baseline separately from the source file, ideally in access-controlled storage.
4. Later, upload the file and that baseline to compare hashes and verify the Ed25519 signature.
5. If the signing key changes, the old baseline will not verify against the new public key; preserve and trust the original public key separately where continuity matters.

### Interpreting results

- Findings are **candidates** unless the evidence explicitly confirms a condition (for example, a hash mismatch against a valid trusted baseline).
- Heuristics can have false positives and false negatives. Validate findings before taking action.
- A CSV/XLSX-only scan cannot prove data poisoning, identify all sensitive-data formats, determine live database permissions, or test server-side vulnerabilities.
- The scanner does not upload raw dataset contents to Nemotron. The optional triage sends only finding metadata and redacted descriptions.
- Scan telemetry stores aggregate counts, extension, hash, and latency—not row contents. The local audit log records the filename and hash; avoid using sensitive filenames.
- Upload safety limits: the read-only scanner rejects files larger than 50 MiB and checks XLSX/XLSM ZIP expansion size/ratio before parsing to reduce accidental resource-exhaustion risk.


## AI Agent Defense Center

The Agent Defenses tab adds a deterministic default-deny tool authorization layer and agent handoff checks to the existing Nemotron workflow. Registered capabilities are limited to dataset schema access, signed-evidence retrieval, read-only SQL, result summarization, and answer verification. Unknown tools and side-effecting actions (such as shell execution, email, file writes, and database writes) are rejected because those capabilities are not implemented. SQL is validated again at the tool-authorization boundary before the isolated SQLite sandbox executes it.

Model outputs crossing selected agent boundaries are checked for configured prompt-injection/jailbreak signals and secret-shaped strings. Retrieved evidence is explicitly labeled as untrusted data and bounded before it is placed in prompts. Agent defense decisions are recorded as privacy-conscious metrics and shown in Plotly charts. The UI includes a policy simulator that only evaluates decisions; it does not execute requested tools.

**Limitations:** these are deterministic heuristics and allowlists, not a complete agent security solution. They can miss attacks and flag benign content. The project does not implement external action tools, production human approval, cloud IAM enforcement, or formal policy verification. Model-based verification is not a security proof. Test the application in an isolated environment and obtain independent security review before any production use.

## YARA artifact scanning (optional)

GradientHarbor includes a read-only YARA Scanner tab for scanning uploaded artifacts against a small built-in heuristic ruleset and optional user-supplied `.yar`/`.yara` rules. Files are read as bytes and never executed. The report includes rule matches, metadata, and a SHA-256 digest; scan events are added to security metrics and the audit trail.

Install the optional engine separately in the project virtual environment:

```powershell
pip install -r requirements-yara.txt
```

The built-in rules are illustrative indicators, not a maintained malware signature corpus. A match is not proof of maliciousness, and no match does not prove safety. Use trusted, maintained rules for broader coverage. The scanner caps artifacts at 25 MiB and custom rule text at 256 KiB.

## Multi-source data security scanning

The **Data Source Scanner** tab supports:

- CSV, XLSX, and XLSM uploads (bounded sample for this tab; the original Vulnerability Investigation tab remains available for full supported-file scans).
- `.sql` / `.txt` SQL script review using SQLGlot. Scripts are statically parsed for non-read-only statements and credential-like assignments; they are **never executed**.
- MySQL and PostgreSQL connections for listing table names and scanning one selected table with a bounded `SELECT ... LIMIT` sample.
- SQLite `.db`, `.sqlite`, and `.sqlite3` uploads scanned from a temporary copy opened with SQLite `mode=ro` and `query_only` enabled.
- MongoDB collection listing and bounded `find` sampling.

Install all dependencies with `pip install -r requirements.txt`. Network database connections require a reachable server and appropriate driver support. Configure database accounts with only the minimum read privileges; the application-level read-only query pattern is not a substitute for least-privilege database grants. Credentials are not included in reports, metrics, or audit events, but are necessarily held transiently in the active Streamlit session while the form is used. Use TLS for remote database connections and never paste production credentials into a shared or untrusted deployment.

Live database scans inspect only the selected table/collection sample (maximum 1,000 rows/documents). They do not audit grants, indexes, server settings, network exposure, or unsampled records. SQL script review is static only. Scanner results are heuristic candidates, not proof of compromise.
