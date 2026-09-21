"""Readable muscle labels (channel code -> "Muscle (side)")."""

LABELS = {
    "L_DELmed": "Deltoid med. (L)",   "R_DELmed": "Deltoid med. (R)",
    "L_BB":     "Biceps (L)",          "R_BB":     "Biceps (R)",
    "L_TBlg":   "Triceps long (L)",    "R_TBlg":   "Triceps long (R)",
    "R_TR":     "Trapezius (R)",
    "L_The_ED_FD_FCR A": "Thenar (L)",              "R_The_ED_FD_FCR A": "Thenar (R)",
    "L_The_ED_FD_FCR B": "Ext. digitorum (L)",      "R_The_ED_FD_FCR B": "Ext. digitorum (R)",
    "L_The_ED_FD_FCR C": "Flex. digitorum (L)",     "R_The_ED_FD_FCR C": "Flex. digitorum (R)",
    "L_The_ED_FD_FCR D": "Flex. carpi rad. (L)",    "R_The_ED_FD_FCR D": "Flex. carpi rad. (R)",
    "Trigger A": "Trigger",
}


def pretty(name):
    """Readable muscle name; falls back to the raw channel code if unmapped."""
    return LABELS.get(name, name)
