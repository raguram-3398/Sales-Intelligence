from pathlib import Path

from sales_intelligence.services.pii_scrubber import (
    scrub_pii,
    count_scrubbed_entities,
)

text = Path("docs/original_transcript.txt").read_text()

scrubbed = scrub_pii(text)

print(scrubbed)
print()
print(count_scrubbed_entities(scrubbed))