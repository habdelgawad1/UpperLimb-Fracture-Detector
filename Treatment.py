def suggest_treatment(bone, injury_type):
    rules = {
        "wrist": {
            "fracture": {
                "Treatment": "Casting or splinting",
                "Medication": "Paracetamol / Ibuprofen",
                "Timeline": "4–6 weeks"
            },
            "normal": {
                "Treatment": "No treatment needed",
                "Medication": "N/A",
                "Timeline": "N/A"
            }
        },
        "hand": {
            "fracture": {
                "Treatment": "Splint or surgery if displaced",
                "Medication": "NSAIDs",
                "Timeline": "3–6 weeks"
            },
            "normal": {
                "Treatment": "No treatment needed",
                "Medication": "N/A",
                "Timeline": "N/A"
            }
        },
        "elbow": {
            "fracture": {
                "Treatment": "Immobilization or ORIF",
                "Medication": "Painkillers, anti-inflammatories",
                "Timeline": "6–12 weeks"
            },
            "normal": {
                "Treatment": "No treatment needed",
                "Medication": "N/A",
                "Timeline": "N/A"
            }
        },
        "shoulder": {
            "fracture": {
                "Treatment": "Sling or surgical fixation",
                "Medication": "Analgesics",
                "Timeline": "6–10 weeks"
            },
            "normal": {
                "Treatment": "No treatment needed",
                "Medication": "N/A",
                "Timeline": "N/A"
            }
        },
        "forearm": {
            "fracture": {
                "Treatment": "Casting or ORIF",
                "Medication": "Paracetamol",
                "Timeline": "6–8 weeks"
            },
            "normal": {
                "Treatment": "No treatment needed",
                "Medication": "N/A",
                "Timeline": "N/A"
            }
        },
        "humerus": {
            "fracture": {
                "Treatment": "Sling or surgical fixation",
                "Medication": "Painkillers",
                "Timeline": "8–12 weeks"
            },
            "normal": {
                "Treatment": "No treatment needed",
                "Medication": "N/A",
                "Timeline": "N/A"
            }
        },
        "finger": {
            "fracture": {
                "Treatment": "Buddy taping or splinting",
                "Medication": "Paracetamol",
                "Timeline": "3–6 weeks"
            },
            "normal": {
                "Treatment": "No treatment needed",
                "Medication": "N/A",
                "Timeline": "N/A"
            }
        }
    }

    try:
        return rules[bone.lower()][injury_type.lower()]
    except KeyError:
        return {
            "error": f"No matching treatment found for {bone} - {injury_type}"
        }