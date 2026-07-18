"""
Vocabulary mappings and synonym dictionaries for the query parser.
"""

COLOR_WORDS = {
    "red", "blue", "white", "black", "yellow", "green", "pink",
    "orange", "purple", "brown", "grey", "gray"
}

GARMENT_WORDS = {
    "shirt", "tshirt", "t-shirt", "tie", "blazer", "hoodie",
    "jacket", "coat", "raincoat", "pants", "jeans", "dress",
    "skirt", "suit", "sweater"
}

SCENE_WORDS = {
    "office", "park", "street", "home", "beach", "restaurant",
    "cafe", "airport", "mall", "forest", "gym"
}

OBJECT_WORDS = {
    "bench", "desk", "laptop", "chair", "table", "tree",
    "car", "bicycle", "umbrella"
}

STYLE_WORDS = {
    "formal", "casual", "business", "sporty", "party"
}

GENDER_MAP = {
    "boy": "male",
    "man": "male",
    "male": "male",
    "gentleman": "male",
    "girl": "female",
    "woman": "female",
    "female": "female",
    "lady": "female"
}

SYNONYMS = {
    "business attire": "formal suit",
    "conference room": "office",
    "workplace": "office",
    "weekend outfit": "casual",
    "button down": "shirt",
    "button-down": "shirt",
    "dress shirt": "shirt",
    "tee": "t-shirt",
    "coat": "jacket"
}
