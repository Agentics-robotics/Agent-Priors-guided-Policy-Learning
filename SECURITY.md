# Security

Report vulnerabilities privately to the maintainers using the repository host's
private vulnerability reporting facility when available. Do not include API keys
or private data in public issues. Rotate an exposed key before requesting removal.

API keys are supplied through environment variables. The public configuration
contains the environment variable name, never its value. Responses transport
receipts redact the configured credential and do not retain request headers.

Generated policies execute through isolated Linux workers using filesystem and
system-call restrictions. Those restrictions are part of the runtime contract;
do not disable them to load an untrusted policy. Model checkpoints use PyTorch
serialization and must be obtained from a trusted source and verified against the
release manifest before loading.

`pixi run audit` checks publication text and symbolic links for common disclosure
patterns. It is a release gate, not a proof that arbitrary external content is safe.
