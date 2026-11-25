# Third-party material

The root MIT license covers PinTrace code and does not replace licenses attached to third-party data or dependencies.

| Material | Source and terms | Attribution in this repository |
| --- | --- | --- |
| Stack Overflow questions and answers | [Stack Overflow content licensing](https://stackoverflow.com/help/licensing). Individual contributions retain their applicable CC BY-SA version. | `data/attribution/stackoverflow.jsonl.gz` lists task IDs, original post URLs, available author records, and reported license labels. Text transformations used for prompts are represented by the exact prompts and source tasks. |
| BigCodeBench v0.1.4 | [Official dataset](https://huggingface.co/datasets/bigcode/bigcodebench), Apache-2.0 | `BIGCODEBENCH-LICENSE`, the supplied dataset card, and original task IDs. |
| PyPI download ranking | [hugovk/top-pypi-packages, release 2025.12](https://github.com/hugovk/top-pypi-packages/releases/tag/2025.12) | Original ranking metadata is preserved in the included JSON. |
| PyPI package metadata | [PyPI JSON API](https://docs.pypi.org/api/json/) | Distribution names, versions, upload dates, artifact URLs, and metadata source fields are preserved. Package distributions are obtained separately under their own licenses when reevaluating programs. |
| OSV advisory records | [OSV data](https://google.github.io/osv.dev/data/) and the originating advisory databases | Original advisory IDs, database-specific metadata, aliases, and source/reference links are preserved. Source advisory terms continue to apply. |
| Model-generated responses | Outputs collected for this study | Model identifiers, exact answer text, prompts, content hashes, and evaluation attribution are retained. These data are not relicensed as third-party software packages by the root MIT notice. |

Stack Overflow attribution metadata covers 1,999 of the 2,000 referenced posts returned by the API. Answer `79854831` was not returned. Its source URL is retained, with unavailable author/license fields represented as null. Questions and answers can have different applicable CC BY-SA versions, so the dataset does not assign one blanket version to all Stack Overflow content.

Python dependencies retain their own licenses. The dependency lock records the distributions used for offline analysis. The repository contains no installed package binaries or model weights.
