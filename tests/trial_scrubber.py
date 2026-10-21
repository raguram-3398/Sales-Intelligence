from pathlib import Path

from sales_intelligence.services.pii_scrubber import (
    count_scrubbed_entities,
    scrub_pii,
)

text = Path("docs/original_transcript.txt").read_text()

scrubbed = scrub_pii(text)

print(scrubbed)
print()
print(count_scrubbed_entities(scrubbed))
