import random

# Letter pools
CONSONANTS = list("bcdfghjklmnpqrstvwxyz")
VOWELS     = list("aeiou")

# Common name-like patterns (C = consonant, V = vowel)
# Lengths are typical for given names
PATTERNS = [
    "CVCV",      # e.g. Lena, Kira
    "CVCVC",     # e.g. Karin, Tomas
    "VCVC",      # e.g. Ario, Elia
    "CVCCV",     # e.g. Marta, Belka
    "VCCV",      # e.g. Anna, Otto (with doubles)
    "CVCVCV",    # e.g. Eliana, Mariko
    "CVC",       # short: Kai, Ren
    "VCV",       # short: Ava, Ira
    "CCVC"
]

def _letter(pool, previous=None, allow_double=True):
    """Pick a letter, optionally allowing a double of the previous one."""
    if previous and allow_double and random.random() < 0.18:
        return previous
    return random.choice(pool)

def generate_name(min_len=3, max_len=7, seed=None) -> str:
    """
    Generate a random name by following a consonant/vowel pattern.
    """
    
    if seed != None: random.seed(seed)
    
    pattern = random.choice(PATTERNS)
    name = []
    prev = None

    for char_type in pattern:
        if char_type == "C":
            letter = _letter(CONSONANTS, prev)
        else:
            letter = _letter(VOWELS, prev)
        name.append(letter)
        prev = letter

    # Capitalize first letter
    return "".join(name).capitalize()

def _make_single_parent_child(parent: str, seed=None) -> str:
    """Helper preserving your original single-parent mutation logic with a length guard."""
    if not parent: 
        return "Aura"
        
    letters = list(parent)
    def is_vowel(c): return c in VOWELS

    strategies = ["swap_letter", "change_ending", "mutate_interior"]
    strategy = random.choice(strategies)

    if strategy == "swap_letter" and len(letters) >= 2:
        idx = random.randint(0, len(letters) - 1)
        candidates = [v for v in VOWELS if v != letters[idx]] if is_vowel(letters[idx]) else [c for c in CONSONANTS if c != letters[idx]]
        if candidates: letters[idx] = random.choice(candidates)

    elif strategy == "change_ending":
        action = random.choice(["shorten", "lengthen", "tweak"])
        if action == "shorten" and len(letters) > 3:
            letters = letters[:-1]
        elif action == "lengthen" and len(letters) < 10: # Guard against endless growth
            last = letters[-1]
            letters.append(random.choice(CONSONANTS if is_vowel(last) else VOWELS))
        else: 
            last = letters[-1]
            candidates = [v for v in VOWELS if v != last] if is_vowel(last) else [c for c in CONSONANTS if last != last]
            if candidates: letters[-1] = random.choice(candidates)

    else:  # mutate_interior
        if len(letters) >= 3:
            idx = random.randint(1, len(letters) - 2)
            c = letters[idx]
            candidates = [v for v in VOWELS if v != c] if is_vowel(c) else [con for con in CONSONANTS if con != c]
            if candidates: letters[idx] = random.choice(candidates)

    # Hard clamp to keep names human-length (max 9 chars)
    if len(letters) > 9:
        letters = letters[:random.randint(4, 8)]

    child = "".join(letters).capitalize()
    if child.lower() == parent:
        last = child[-1].lower()
        candidates = [v for v in VOWELS if v != last] if last in VOWELS else [c for c in CONSONANTS if last != c]
        if candidates:
            child = child[:-1] + random.choice(candidates)
            
    return child

def make_child(parentA: str, parentB: str = None, seed=None) -> str:
    """
    Create a name related to one or two parents, safely clamped to avoid string bloat.
    """
    if seed is not None:
        random.seed(seed)
        
    parentA = parentA.lower() if parentA else ""
    
    if not parentB:
        return _make_single_parent_child(parentA, seed)
        
    parentB = parentB.lower()
    def is_vowel(c): return c in VOWELS

    strategies = ["splice", "interleave", "blend_halves"]
    strategy = random.choice(strategies)
    
    if strategy == "splice":
        splitA = random.randint(1, max(1, len(parentA) - 1))
        splitB = random.randint(0, max(0, len(parentB) - 1))
        letters = list(parentA[:splitA] + parentB[splitB:])
        
    elif strategy == "interleave":
        # Instead of growing infinitely, we sample/zip up to a max length
        letters = []
        max_len = random.randint(3, 8)
        i, j = 0, 0
        while (i < len(parentA) or j < len(parentB)) and len(letters) < max_len:
            if i < len(parentA) and (random.random() < 0.5 or j >= len(parentB)):
                letters.append(parentA[i])
                i += 1
            elif j < len(parentB):
                letters.append(parentB[j])
                j += 1
                
    else:  # blend_halves
        midA = max(1, len(parentA) // 2)
        midB = len(parentB) // 2
        letters = list(parentA[:midA] + parentB[midB:])
        
    if not letters:
        letters = list(parentA)

    # Hard clamp to ensure children never exceed 9 characters
    if len(letters) > 9:
        max_cut = random.randint(4, 8)
        letters = letters[:max_cut]

    # Apply a light interior or letter mutation
    if len(letters) >= 2 and random.random() < 0.7:
        idx = random.randint(0, len(letters) - 1)
        c = letters[idx]
        candidates = [v for v in VOWELS if v != c] if is_vowel(c) else [con for con in CONSONANTS if con != c]
        if candidates:
            letters[idx] = random.choice(candidates)

    child = "".join(letters).capitalize()

    if child.lower() == parentA or child.lower() == parentB:
        last = child[-1].lower()
        candidates = [v for v in VOWELS if v != last] if last in VOWELS else [c for c in CONSONANTS if last != c]
        if candidates:
            child = child[:-1] + random.choice(candidates)

    return child