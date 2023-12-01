# Subset of classes for system 2
DATASET = "Babelscape/multinerd"
SUBSET_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14]
MAP_SYS_B = {
    elem: ix 
    for ix, elem in enumerate(SUBSET_CLASSES)
}
LABELS_SYS_A = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-ANIM",
    "I-ANIM",
    "B-BIO",
    "I-BIO",
    "B-CEL",
    "I-CEL",
    "B-DIS",
    "I-DIS",
    "B-EVE",
    "I-EVE",
    "B-FOOD",
    "I-FOOD",
    "B-INST",
    "I-INST",
    "B-MEDIA",
    "I-MEDIA",
    "B-MYTH",
    "I-MYTH",
    "B-PLANT",
    "I-PLANT",
    "B-TIME",
    "I-TIME",
    "B-VEHI",
    "I-VEHI",
]

LABELS_SYS_B = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-ANIM",
    "I-ANIM",
    "B-DIS",
    "I-DIS",
]


# Function to pass on to map method
def filter_subclasses(exm):
    tmp = exm
    tmp["ner_tags"] = [MAP_SYS_B[ix] if ix in SUBSET_CLASSES else 0 for ix in exm["ner_tags"]]
    return tmp
