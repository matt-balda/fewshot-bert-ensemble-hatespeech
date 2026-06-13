"""
augment/templates.py — Etapa 2: Templates Semânticos por Categoria

Define templates de prompt que preservam:
  • Intenção semântica
  • Polaridade
  • Tipo de ataque
  • Contexto linguístico

Cada template é parametrizável por categoria e estilo de variação.
"""

from typing import List

# ---------------------------------------------------------------------------
# Style dimensions (Etapa 5 — Diversidade Linguística)
# ---------------------------------------------------------------------------

TONE_VARIANTS = ["aggressive", "passive-aggressive", "sarcastic", "cold", "direct"]
LENGTH_VARIANTS = ["very short (under 10 words)", "short (10-20 words)", "medium (20-40 words)"]
FORMALITY_VARIANTS = ["very informal", "informal", "semi-formal"]
SLANG_VARIANTS = ["heavy slang", "some slang", "no slang"]
TYPO_VARIANTS = ["some intentional typos", "no typos"]
SYNTAX_VARIANTS = [
    "simple sentences",
    "fragmented sentences",
    "rhetorical questions",
    "imperative sentences",
]

# ---------------------------------------------------------------------------
# Base prompt template — filled per cluster
# ---------------------------------------------------------------------------

BASE_PROMPT = """You are assisting in building a hate speech detection academic benchmark dataset.
Your task is to generate {n_variants} diverse, realistic examples of {category} hate speech in English,
inspired by the real examples below.

Each generated example should:
- Be written in Twitter/social media style (informal, possibly with typos)
- Preserve the semantic intent: {intent}
- Vary in: tone ({tone}), length ({length}), formality ({formality}), vocabulary
- NOT include hashtags, mentions, or URLs

Real examples from this category:
{examples}

Generate exactly {n_variants} NEW, distinct examples. Output ONLY the examples, one per line.
Do NOT number them. Do NOT include quotes. Do NOT apologize or add explanations."""


# ---------------------------------------------------------------------------
# Category-specific intents and keywords
# ---------------------------------------------------------------------------

CATEGORY_INTENTS = {
    "racial_attacks": {
        "intent": "attacking or denigrating people based on their race or ethnicity",
        "keywords": ["racial slurs", "ethnic stereotypes", "racial inferiority"],
    },
    "xenophobia": {
        "intent": "hostility toward immigrants or foreigners",
        "keywords": ["anti-immigrant", "nationalistic hostility", "anti-foreign"],
    },
    "homophobia": {
        "intent": "hostility or derogation targeting LGBTQ+ people",
        "keywords": ["anti-gay slurs", "homophobic stereotypes"],
    },
    "sexism": {
        "intent": "denigrating or objectifying people based on gender",
        "keywords": ["misogynistic", "sexist stereotypes", "gender-based attacks"],
    },
    "religious_intolerance": {
        "intent": "attacking people based on religion or religious practice",
        "keywords": ["anti-religious slurs", "religious stereotypes"],
    },
    "other": {
        "intent": "expressing general dehumanizing or hateful language",
        "keywords": ["general slurs", "dehumanizing language"],
    },
}


def build_prompt(
    category: str,
    few_shot_examples: List[str],
    n_variants: int = 20,
    tone: str = "aggressive",
    length: str = "short (10-20 words)",
    formality: str = "very informal",
) -> str:
    """
    Build a complete few-shot prompt for a given semantic category and style.

    Parameters
    ----------
    category        : One of the keys in CATEGORY_INTENTS.
    few_shot_examples : Real examples from this cluster.
    n_variants      : Number of examples to generate in this call.
    tone / length / formality : Style dimensions for Etapa 5.
    """
    cat_info = CATEGORY_INTENTS.get(category, CATEGORY_INTENTS["other"])
    examples_str = "\n".join(f"- {ex}" for ex in few_shot_examples)

    return BASE_PROMPT.format(
        n_variants=n_variants,
        category=category.replace("_", " "),
        intent=cat_info["intent"],
        tone=tone,
        length=length,
        formality=formality,
        examples=examples_str,
    )


def style_combinations(n: int = 20) -> List[dict]:
    """
    Return up to n style combinations by cycling through the style dimensions.
    Used to guarantee linguistic diversity in Etapa 5.
    """
    import itertools
    combos = list(itertools.product(
        TONE_VARIANTS,
        LENGTH_VARIANTS,
        FORMALITY_VARIANTS,
        TYPO_VARIANTS,
        SYNTAX_VARIANTS,
    ))
    # Subsample deterministically
    step = max(1, len(combos) // n)
    selected = combos[::step][:n]
    return [
        {
            "tone": c[0], "length": c[1], "formality": c[2],
            "typos": c[3], "syntax": c[4],
        }
        for c in selected
    ]


# ---------------------------------------------------------------------------
# Neither (neutral/non-offensive) category intents and prompt
# ---------------------------------------------------------------------------

NEITHER_CATEGORY_INTENTS = {
    "sarcasm_humor": {
        "intent": "expressing sarcasm, irony, or humor in a non-offensive, non-hateful way",
        "keywords": ["sarcasm", "irony", "jokes", "wit", "humor", "satire"],
    },
    "political_neutral": {
        "intent": "neutral political commentary or opinion without attacking any group",
        "keywords": ["political opinion", "commentary", "debate", "policy", "election"],
    },
    "everyday_talk": {
        "intent": "casual everyday conversation, personal observations, or feelings",
        "keywords": ["daily life", "casual talk", "personal", "feelings", "observations"],
    },
    "news_media": {
        "intent": "sharing or commenting on news or current events neutrally",
        "keywords": ["news", "current events", "journalism", "media", "reporting"],
    },
    "sports_entertainment": {
        "intent": "talking about sports, music, movies, or pop culture",
        "keywords": ["sports", "music", "movies", "celebrities", "entertainment"],
    },
    "other_neutral": {
        "intent": "general non-offensive social media content",
        "keywords": ["general content", "neutral", "non-offensive", "everyday"],
    },
}

NEITHER_BASE_PROMPT = """You are assisting in building a text classification benchmark dataset.
Your task is to generate {n_variants} diverse, realistic examples of NON-OFFENSIVE, NON-HATEFUL social media content in English,
inspired by the real examples below.

Each generated example should:
- Be written in Twitter/social media style (informal, casual)
- Be clearly non-offensive, non-hateful, and non-threatening
- Preserve the type of content: {intent}
- Vary in: tone ({tone}), length ({length}), formality ({formality})
- NOT include hashtags, mentions, or URLs

Real examples from this category:
{examples}

Generate exactly {n_variants} NEW, distinct examples. Output ONLY the examples, one per line.
Do NOT number them. Do NOT include quotes. Do NOT add explanations."""


def build_neither_prompt(
    category: str,
    few_shot_examples: List[str],
    n_variants: int = 20,
    tone: str = "casual",
    length: str = "short (10-20 words)",
    formality: str = "informal",
) -> str:
    """
    Build a few-shot prompt for generating neutral/non-offensive content.

    Parameters
    ----------
    category          : One of the keys in NEITHER_CATEGORY_INTENTS.
    few_shot_examples : Real 'neither' examples from this cluster.
    n_variants        : Number of examples to generate in this call.
    tone / length / formality : Style dimensions for Etapa 5.
    """
    cat_info     = NEITHER_CATEGORY_INTENTS.get(category, NEITHER_CATEGORY_INTENTS["other_neutral"])
    examples_str = "\n".join(f"- {ex}" for ex in few_shot_examples)

    return NEITHER_BASE_PROMPT.format(
        n_variants=n_variants,
        category=category.replace("_", " "),
        intent=cat_info["intent"],
        tone=tone,
        length=length,
        formality=formality,
        examples=examples_str,
    )

