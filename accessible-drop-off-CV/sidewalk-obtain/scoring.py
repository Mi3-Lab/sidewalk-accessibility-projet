# Passability scores based on the "Accessibility for Whom?" paper's findings,
# but adapted to use the median_severity from the JSON data.
# A score of 1.0 is 100% passable, and 0.0 is 0% passable.
SEVERITY_SCORES = {
    1: 0.9, # Very low severity (e.g., minor bump, great curb ramp)
    2: 0.6, # Medium severity (e.g., standard curb ramp, small crack)
    3: 0.3, # High severity (e.g., significant obstacle, missing curb ramp)
    4: 0.1, # Critical severity (e.g., blocked sidewalk)
}

# Adjustments for different mobility aids,
# based on the relative difficulty of navigating different label types.
MOBILITY_AID_ADJUSTMENTS = {
    "CurbRamp": {
        "Cane": 0.05,
        "Walker": -0.05,
        "Mobility Scooter": -0.2,
        "Manual Wheelchair": -0.15,
        "Motorized Wheelchair": -0.1
    },
    "NoCurbRamp": {
        "Cane": 0.0,
        "Walker": -0.2,
        "Mobility Scooter": -0.6,
        "Manual Wheelchair": -0.7,
        "Motorized Wheelchair": -0.8
    },
    "NoSidewalk": {
        "Cane": 0.0,
        "Walker": -0.1,
        "Mobility Scooter": -0.3,
        "Manual Wheelchair": -0.4,
        "Motorized Wheelchair": -0.5
    },
    "SurfaceProblem": {
        "Cane": 0.0,
        "Walker": -0.1,
        "Mobility Scooter": -0.2,
        "Manual Wheelchair": -0.3,
        "Motorized Wheelchair": -0.3
    },
    "Obstacle": {
        "Cane": 0.0,
        "Walker": -0.1,
        "Mobility Scooter": -0.3,
        "Manual Wheelchair": -0.4,
        "Motorized Wheelchair": -0.5
    },
    "Crosswalk": {
        "Cane": 0.05,
        "Walker": 0.05,
        "Mobility Scooter": 0.0,
        "Manual Wheelchair": 0.0,
        "Motorized Wheelchair": 0.0
    },
    "Signal": {
        "Cane": 0.05,
        "Walker": 0.05,
        "Mobility Scooter": 0.0,
        "Manual Wheelchair": 0.0,
        "Motorized Wheelchair": 0.0
    },
}

def calculate_passability_score(spot_data, mobility_aid):
    """
    Calculates the passability score for a spot based on JSON data attributes.
    """
    label_type = spot_data.get("labels", [])[0] if spot_data.get("labels") else None
    
    if not label_type:
        return 1.0 # Return a perfect score if no label is found

    # Get median severity and ensure it's a valid key
    median_severity = spot_data.get("median_severity")
    if median_severity is None:
        return 1.0 # If severity is null, assume no issue and return a high score
    
    base_score = SEVERITY_SCORES.get(median_severity, 0.5)
    
    # Apply mobility aid adjustment
    adjustment = MOBILITY_AID_ADJUSTMENTS.get(label_type, {}).get(mobility_aid, 0.0)
    adjusted_score = base_score + adjustment

    # Ensure the score is within the valid range [0.0, 1.0]
    final_score = max(0.0, min(1.0, adjusted_score))

    return final_score