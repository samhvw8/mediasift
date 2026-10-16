# Shared brief for gatherers (framesift model alternatives, 2026-10-16)

framesift = local video/audio "sensor" CLI for agents. Python 3.14. Main machine: Apple Silicon M1 Pro 16 GB Mac. Also an RTX 3060 Linux box over ssh.

Current stack and measured numbers (do NOT re-research; compare against them):
- Vietnamese ASR: QUAN-Transcribe-1.0 (huggingface qnguyen3/QUAN-Transcribe-1.0, created 2026-10-14; EmbeddingGemma 2 audio encoder + 6-layer CTC reader, 356M, Core ML fp16 via pyobjc). M1 Pro: 156x realtime. Output lowercase, no punctuation, numbers as words. Clean 47-min Vietnamese dub: WER 5.45% vs mlx-whisper large-v3-turbo 2.75% (QUAN drops fillers and English names like ChatGPT/AGI). Author's benchmark avg 6.87% vs Qwen3-ASR-1.7B 7.01%, PhoWhisper-large 7.21%.
- Other languages: parakeet-tdt-0.6b-v3 via parakeet-mlx (~25x realtime M1 Pro); mlx-whisper large-v3-turbo for the rest; faster-whisper on CUDA.
- English EmbeddingGemma 2 reader (mlboydaisuke eg2-transcribe): rejected, 4.21% LibriSpeech test-clean, fixed 200k-word vocab.
- Language ID: whisper-tiny on CPU (faster-whisper) over 30 s.
- Sound tagging: CED-small int8 (22 MB) via sherpa-onnx CPU, 3 s windows (3 min audio in 4.8-7.4 s). onnxruntime CoreML EP was slower (18.8 s).
- Speakers: TitaNet-small (sherpa-onnx), greedy clustering.
- OCR: Apple Vision accurate. Search: EmbeddingGemma 2 (google/embeddinggemma-2, released 2026-09-14, apache-2.0; text+image+audio embeddings) via llama.cpp llama-server.

Today is 2026-10-16. Never put a year in a search query.

For every candidate record: what it is, license, size, measured speed on Apple Silicon (which chip) or other HW, accuracy numbers WITH test set, maturity (GitHub stars, last commit / HF lastModified, downloads), and whether it runs from Python 3.14 (pip wheel? pyobjc? onnx?) or as a CLI. Flag claims that only come from social posts as UNVERIFIED.

Tools: Parallax MCP (load with ToolSearch "select:mcp__parallax__web_search,mcp__parallax__batch_fetch,mcp__parallax__fetch_page,mcp__parallax__social_search,mcp__parallax__twitter_search,mcp__parallax__search_reddit"). Hugging Face API via curl: `curl -s "https://huggingface.co/api/models?search=<q>&sort=lastModified&limit=50"` and `https://huggingface.co/api/models/<id>` and model cards at `https://huggingface.co/<id>/raw/main/README.md`. GitHub: `gh repo view <owner/repo> --json stargazerCount,pushedAt,licenseInfo,description`, `gh release list -R <repo>`, `gh issue view`.
