# Public dataset and report notes

This release contains a pseudonymized copy of the course material, prepared for public access.

- The 13,348 observations, label columns, 1024-dimensional embeddings and 46 annotator groups are retained.
- Original record identifiers were replaced with sequential public record labels.
- Original annotator identifiers were replaced with new numeric group labels. No correspondence table is included. Group membership is preserved for the annotator-separated split.
- Email addresses in source-paper text were redacted in 30 paragraphs.
- The rendered report's annotator tables use the replacement groups; source dataframe row indices are also replaced. Other original report content is retained.
- The public repository has a fresh Git history. The unprocessed dataset and original report are not present in earlier public commits.

Embeddings and model weights are unchanged. Email redaction changes paragraph length for the affected rows, so a small part of the numeric feature input differs from the historical training data. The archived report metrics therefore describe the original study, not a new fit on this public copy. Full training has not been rerun. The original dataset/report are retained privately.
