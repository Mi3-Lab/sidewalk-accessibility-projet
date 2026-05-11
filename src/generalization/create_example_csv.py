#!/usr/bin/env python3
"""
Create sample test_images.csv template to show you what it should look like.
"""

import pandas as pd
from pathlib import Path

# Create example data
example_data = {
    "image_id": ["dc_001", "dc_002", "col_001", "col_002", "col_003"],
    "image_path": [
        "data/generalization/images/dc_curbramp_001.jpg",
        "data/generalization/images/dc_nocurb_001.jpg",
        "data/generalization/images/columbus_curb_001.jpg",
        "data/generalization/images/columbus_nocurb_001.jpg",
        "data/generalization/images/columbus_unmarked_001.jpg",
    ],
    "ps_label": ["CurbRamp", "NoCurbRamp", "CurbRamp", "NoCurbRamp", "Unmarked"],
    "city": ["Washington DC", "Washington DC", "Columbus OH", "Columbus OH", "Columbus OH"],
    "lat": [38.8951, 38.8952, 39.9612, 39.9613, 39.9614],
    "lon": [-77.0369, -77.0370, -82.9988, -82.9989, -82.9990],
    "source": ["PSW", "PSW", "PSW", "PSW", "PSW"],
    "notes": ["Accessible ramp", "No curb cut", "Good ramp", "Broken sidewalk", "Survey data"],
}

df = pd.DataFrame(example_data)

output_path = Path("data/generalization/test_images_example.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print(f"✅ Example CSV created: {output_path}")
print("\nFormat:")
print(df.to_string())
print(f"\nUse this as template for your real test_images.csv")
