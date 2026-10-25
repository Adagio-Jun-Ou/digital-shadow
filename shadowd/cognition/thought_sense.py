"""
ThoughtSense — Intent, Emotion, and Subtext Detection

Analyzes text to detect:
- Intent (what the person wants to achieve)
- Emotion (underlying emotional state)
- Subtext (what they're really saying beyond literal words)
- Cognitive load (how complex/heavy the communication is)
- Urgency (time sensitivity)
"""

import copy
import logging
import re
import threading
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class Intent(Enum):
    INFORM = "inform"
    REQUEST = "request"
    COMMIT = "commit"
    DECLINE = "decline"
    QUESTION = "question"
    DIRECTIVE = "directive"
    SOCIAL = "social"
    EMOTIONAL = "emotional"
    CORRECTION = "correction"
    WARNING = "warning"


class Emotion(Enum):
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    ANTICIPATION = "anticipation"
    TRUST = "trust"


class SubtextType(Enum):
    HESITATION = "hesitation"
    URGENCY = "urgency"
    AVOIDANCE = "avoidance"
    INDIRECT_REQUEST = "indirect_request"
    CONCERN = "concern"
    ENTHUSIASM = "enthusiasm"
    PASSIVE_AGGRESSIVE = "passive_aggressive"
    AUTHORITY_ASSERTION = "authority_assertion"


class UrgencyLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class IntentResult:
    intent: str
    confidence: float
    keywords_matched: List[str]

    def to_dict(self) -> Dict:
        return {
            "intent": self.intent,
            "confidence": round(self.confidence, 3),
            "keywords_matched": list(self.keywords_matched),
        }


@dataclass
class EmotionResult:
    emotion: str
    intensity: float
    keywords_matched: List[str]

    def to_dict(self) -> Dict:
        return {
            "emotion": self.emotion,
            "intensity": round(self.intensity, 3),
            "keywords_matched": list(self.keywords_matched),
        }


@dataclass
class SubtextResult:
    subtext_type: str
    confidence: float
    evidence: List[str]

    def to_dict(self) -> Dict:
        return {
            "subtext_type": self.subtext_type,
            "confidence": round(self.confidence, 3),
            "evidence": list(self.evidence),
        }


@dataclass
class ThoughtAnalysis:
    text: str
    intents: List[IntentResult]
    emotions: List[EmotionResult]
    subtexts: List[SubtextResult]
    cognitive_load: float
    urgency: str
    urgency_confidence: float
    timestamp: str
    summary: str

    def to_dict(self) -> Dict:
        return {
            "text": self.text[:_MAX_TEXT_LENGTH],
            "intents": [i.to_dict() for i in self.intents],
            "emotions": [e.to_dict() for e in self.emotions],
            "subtexts": [s.to_dict() for s in self.subtexts],
            "cognitive_load": round(self.cognitive_load, 3),
            "urgency": self.urgency,
            "urgency_confidence": round(self.urgency_confidence, 3),
            "timestamp": self.timestamp,
            "summary": self.summary,
        }


_INTENT_PATTERNS: Dict[str, List[Tuple[str, float]]] = {
    Intent.INFORM.value: [
        (r'\bjust so you know\b', 0.8),
        (r'\bFYI\b', 0.9),
        (r'\bnote that\b', 0.7),
        (r'\bbe aware\b', 0.7),
        (r'\bI want to tell you\b', 0.7),
        (r'\blet me inform\b', 0.8),
        (r'\bfyi\b', 0.85),
        (r'\bfor your information\b', 0.85),
        (r'\bannouncing\b', 0.7),
        (r'\breported\b', 0.6),
        (r'\bsharing\b', 0.5),
        (r'\bI found\b', 0.5),
        (r'\bdiscovered\b', 0.6),
    ],
    Intent.REQUEST.value: [
        (r'\bcan you\b', 0.7),
        (r'\bcould you\b', 0.7),
        (r'\bplease\b', 0.8),
        (r'\bI need\b', 0.65),
        (r'\bwould you\b', 0.6),
        (r'\bhelp me\b', 0.8),
        (r'\bI want\b', 0.5),
        (r'\bget me\b', 0.7),
        (r'\bI\'d like\b', 0.65),
        (r'\bwould you mind\b', 0.7),
        (r'\bis it possible\b', 0.6),
        (r'\bdo you have\b', 0.5),
        (r'\bmay I\b', 0.65),
        (r'\brequest\b', 0.7),
    ],
    Intent.COMMIT.value: [
        (r'\bI will\b', 0.85),
        (r'\bI promise\b', 0.9),
        (r'\bconsider it done\b', 0.9),
        (r'\bI\'ll take care\b', 0.75),
        (r'\byou can count on me\b', 0.9),
        (r'\bI\'m going to\b', 0.65),
        (r'\bI commit\b', 0.9),
        (r'\bI guarantee\b', 0.85),
        (r'\bdone deal\b', 0.8),
        (r'\bcount on it\b', 0.85),
        (r'\bsure thing\b', 0.6),
        (r'\bI\'m on it\b', 0.75),
    ],
    Intent.DECLINE.value: [
        (r'\bno\b', 0.5),
        (r'\bI can\'t\b', 0.8),
        (r'\bnot possible\b', 0.8),
        (r'\bdeclined?\b', 0.85),
        (r'\brefuse\b', 0.85),
        (r'\bnot interested\b', 0.8),
        (r'\bI won\'t\b', 0.8),
        (r'\bunfortunately\b', 0.6),
        (r'\bwe cannot\b', 0.75),
        (r'\bnot available\b', 0.7),
        (r'\bI must decline\b', 0.9),
        (r'\bI decline\b', 0.95),
        (r'\bnot going to happen\b', 0.85),
    ],
    Intent.QUESTION.value: [
        (r'\bwhat\b', 0.5),
        (r'\bhow\b', 0.5),
        (r'\bwhy\b', 0.6),
        (r'\bwhen\b', 0.5),
        (r'\bwhere\b', 0.5),
        (r'\bwho\b', 0.5),
        (r'\b\?', 0.4),
        (r'\bdid you\b', 0.6),
        (r'\bdo you know\b', 0.7),
        (r'\bcan you tell me\b', 0.75),
        (r'\bI wonder\b', 0.55),
        (r'\bcurious\b', 0.5),
    ],
    Intent.DIRECTIVE.value: [
        (r'\byou must\b', 0.85),
        (r'\byou need to\b', 0.75),
        (r'\byou should\b', 0.65),
        (r'\bmake sure\b', 0.7),
        (r'\bensures?\b', 0.7),
        (r'\bdo not\b', 0.6),
        (r'\bdon\'t\b', 0.5),
        (r'\brequired\b', 0.7),
        (r'\bmandatory\b', 0.85),
        (r'\bmust\b', 0.7),
        (r'\bimperative\b', 0.85),
        (r'\bno choice\b', 0.75),
    ],
    Intent.SOCIAL.value: [
        (r'\bhello\b', 0.7),
        (r'\bhi\b', 0.6),
        (r'\bhey\b', 0.5),
        (r'\bthanks\b', 0.7),
        (r'\bthank you\b', 0.75),
        (r'\bcongratulations\b', 0.8),
        (r'\bcongrats\b', 0.75),
        (r'\bgood morning\b', 0.75),
        (r'\bgood evening\b', 0.75),
        (r'\bhow are you\b', 0.8),
        (r'\bnice to meet\b', 0.8),
        (r'\bappreciate\b', 0.65),
        (r'\bwelcome\b', 0.6),
        (r'\bcheers\b', 0.5),
    ],
    Intent.EMOTIONAL.value: [
        (r'\bI feel\b', 0.8),
        (r'\bI\'m sad\b', 0.85),
        (r'\bI\'m happy\b', 0.8),
        (r'\bI\'m angry\b', 0.85),
        (r'\bI\'m frustrated\b', 0.8),
        (r'\bI\'m excited\b', 0.75),
        (r'\bI love\b', 0.7),
        (r'\bI hate\b', 0.8),
        (r'\bI\'m worried\b', 0.7),
        (r'\bit hurts\b', 0.75),
        (r'\bI\'m overwhelmed\b', 0.7),
        (r'\bI\'m scared\b', 0.8),
        (r'\bI feel like\b', 0.65),
    ],
    Intent.CORRECTION.value: [
        (r'\bactually\b', 0.7),
        (r'\bno,?\b', 0.5),
        (r'\bthat\'s wrong\b', 0.85),
        (r'\bthat\'s incorrect\b', 0.85),
        (r'\bI think you mean\b', 0.8),
        (r'\bcorrection\b', 0.85),
        (r'\bI disagree\b', 0.7),
        (r'\bnot quite\b', 0.7),
        (r'\bthat\'s not right\b', 0.8),
        (r'\blet me correct\b', 0.85),
        (r'\bmisspoke\b', 0.7),
    ],
    Intent.WARNING.value: [
        (r'\bbe careful\b', 0.8),
        (r'\bwatch out\b', 0.85),
        (r'\bwarning\b', 0.9),
        (r'\bdanger\b', 0.9),
        (r'\brisk\b', 0.6),
        (r'\bbeware\b', 0.85),
        (r'\bcaution\b', 0.8),
        (r'\bhazard\b', 0.85),
        (r'\bheads up\b', 0.7),
        (r'\bdon\'t do that\b', 0.75),
        (r'\blooks risky\b', 0.7),
        (r'\bcould be dangerous\b', 0.8),
    ],
}

_EMOTION_KEYWORDS: Dict[str, Dict[str, List[str]]] = {
    Emotion.JOY.value: {
        "strong": ["thrilled", "ecstatic", "elated", "overjoyed", "euphoric", "delighted", "wonderful"],
        "moderate": ["happy", "glad", "pleased", "cheerful", "excited", "great", "fantastic", "awesome", "excellent"],
        "mild": ["okay", "fine", "good", "nice", "pleasant", "content", "satisfied"],
    },
    Emotion.SADNESS.value: {
        "strong": ["devastated", "heartbroken", "devastating", "despondent", "grief", "miserable", "despair"],
        "moderate": ["sad", "unhappy", "depressed", "disappointed", "down", "blue", "upset", "hurt", "sorrow"],
        "mild": ["melancholy", "wistful", "somber", "gloomy", "pensive", "meh"],
    },
    Emotion.ANGER.value: {
        "strong": ["furious", "enraged", "livid", "incensed", "seething", "outraged", "infuriated"],
        "moderate": ["angry", "mad", "annoyed", "frustrated", "irritated", "pissed", "aggravated", "hostile"],
        "mild": ["bothered", "irked", "perturbed", "agitated", "impatient", "cross"],
    },
    Emotion.FEAR.value: {
        "strong": ["terrified", "petrified", "panicked", "horrified", "dread", "phobic", "paralyzed"],
        "moderate": ["afraid", "scared", "anxious", "worried", "nervous", "fearful", "uneasy"],
        "mild": ["concerned", "cautious", "hesitant", "apprehensive", "unsettled", "wary"],
    },
    Emotion.SURPRISE.value: {
        "strong": ["shocked", "astounded", "flabbergasted", "dumbfounded", "stunned", "incredulous"],
        "moderate": ["surprised", "amazed", "astonished", "unexpected", "startled", "taken aback"],
        "mild": ["unexpectedly", "curious", "oh", "huh", "interesting", "well well"],
    },
    Emotion.DISGUST.value: {
        "strong": ["revolted", "repulsed", "nauseated", "sickened", "appalled", "horrified", "repugnant"],
        "moderate": ["disgusted", "gross", "nasty", "repulsive", "vile", "revolting", "awful"],
        "mild": ["unpleasant", "icky", "distasteful", "offensive", "off-putting", "yuck"],
    },
    Emotion.ANTICIPATION.value: {
        "strong": ["eagerly", "impatiently", "can\'t wait", "countdown", "dying to", "itching to"],
        "moderate": ["looking forward", "anticipating", "excited", "expecting", "upcoming", "soon"],
        "mild": ["hope", "wonder", "curious", "perhaps", "maybe", "might", "possible"],
    },
    Emotion.TRUST.value: {
        "strong": ["absolutely", "undoubtedly", "unwavering", "completely trust", "full confidence", "rely on completely"],
        "moderate": ["trust", "believe", "confident", "rely", "faith", "certain", "sure"],
        "mild": ["seem", "appear", "likely", "probably", "assume", "gather", "suppose"],
    },
}

_SUBTEXT_PATTERNS: Dict[str, List[Tuple[str, float]]] = {
    SubtextType.HESITATION.value: [
        (r'\bmaybe\b', 0.6),
        (r'\bsort of\b', 0.7),
        (r'\bkind of\b', 0.7),
        (r'\bI think\b', 0.5),
        (r'\bperhaps\b', 0.6),
        (r'\bsorta\b', 0.7),
        (r'\bkinda\b', 0.7),
        (r'\bpossibly\b', 0.5),
        (r'\bI guess\b', 0.6),
        (r'\bI suppose\b', 0.6),
        (r'\bit seems\b', 0.5),
        (r'\bsomewhat\b', 0.55),
        (r'\bnot really\b', 0.65),
        (r'\balmost\b', 0.4),
    ],
    SubtextType.URGENCY.value: [
        (r'\bASAP\b', 0.9),
        (r'\basap\b', 0.85),
        (r'\burgent\b', 0.9),
        (r'\bemergency\b', 0.95),
        (r'\bdeadline\b', 0.75),
        (r'\bneed by\b', 0.8),
        (r'\bright now\b', 0.85),
        (r'\bimmediately\b', 0.85),
        (r'\btime sensitive\b', 0.8),
        (r'\bcritical\b', 0.8),
        (r'\bcrucial\b', 0.7),
        (r'\bpressing\b', 0.75),
        (r'\bno time\b', 0.8),
        (r'\bbefore\s+\w+\s+(?:end|close|deadline)\b', 0.7),
    ],
    SubtextType.AVOIDANCE.value: [
        (r'\blet\'s not\b', 0.8),
        (r'\bnot ready\b', 0.75),
        (r'\banother time\b', 0.7),
        (r'\bmaybe later\b', 0.7),
        (r'\bI\'d rather not\b', 0.8),
        (r'\bnot now\b', 0.65),
        (r'\bwe can talk about it later\b', 0.7),
        (r'\bpass\b', 0.6),
        (r'\bI\'ll pass\b', 0.7),
        (r'\bnot a good time\b', 0.75),
        (r'\bnot the right time\b', 0.75),
        (r'\bput it off\b', 0.7),
        (r'\brain check\b', 0.7),
    ],
    SubtextType.INDIRECT_REQUEST.value: [
        (r'\bwould be great if\b', 0.8),
        (r'\bcould you possibly\b', 0.8),
        (r'\bit would help if\b', 0.7),
        (r'\bI was hoping\b', 0.7),
        (r'\bif you could\b', 0.65),
        (r'\bit\'d be nice\b', 0.6),
        (r'\bmight want to\b', 0.55),
        (r'\bwould appreciate\b', 0.75),
        (r'\bdo you think you could\b', 0.75),
        (r'\bif it\'s not too much\b', 0.8),
        (r'\bwhen you get a chance\b', 0.6),
        (r'\bat your convenience\b', 0.6),
    ],
    SubtextType.CONCERN.value: [
        (r'\bI hope\b', 0.6),
        (r'\bworried about\b', 0.75),
        (r'\bconcerned\b', 0.75),
        (r'\bI\'m afraid\b', 0.7),
        (r'\bhopefully\b', 0.5),
        (r'\bI wonder if\b', 0.55),
        (r'\bfingers crossed\b', 0.6),
        (r'\bbut what if\b', 0.6),
        (r'\bis it safe\b', 0.65),
        (r'\bmake sure\b', 0.5),
        (r'\bI just want to be sure\b', 0.7),
        (r'\bhope nothing goes wrong\b', 0.7),
    ],
    SubtextType.ENTHUSIASM.value: [
        (r'!+', 0.6),
        (r'\bamazing\b', 0.75),
        (r'\bincredible\b', 0.8),
        (r'\bfantastic\b', 0.75),
        (r'\bawesome\b', 0.7),
        (r'\blove it\b', 0.8),
        (r'\bcan\'t wait\b', 0.75),
        (r'\bso excited\b', 0.85),
        (r'\babsolutely\b', 0.55),
        (r'\bdying to\b', 0.75),
        (r'\bthis is the best\b', 0.8),
        (r'\bthrilled\b', 0.8),
        (r'\bwow\b', 0.65),
        (r'\bwoohoo\b', 0.7),
    ],
    SubtextType.PASSIVE_AGGRESSIVE.value: [
        (r'\bfine\b', 0.55),
        (r'\bwhatever\b', 0.7),
        (r'\bif you must\b', 0.8),
        (r'\bI suppose\b', 0.5),
        (r'\bas usual\b', 0.65),
        (r'\bof course\b', 0.4),
        (r'\bsure\b', 0.35),
        (r'\bno worries\b', 0.35),
        (r'\bthanks anyway\b', 0.65),
        (r'\bI guess\b', 0.45),
        (r'\bper your request\b', 0.6),
        (r'\bnoted\b', 0.45),
        (r'\bI\'ll just\b', 0.6),
        (r'\bclearly\b', 0.4),
    ],
    SubtextType.AUTHORITY_ASSERTION.value: [
        (r'\bI need\b', 0.6),
        (r'\byou must\b', 0.8),
        (r'\bno choice\b', 0.8),
        (r'\bI demand\b', 0.85),
        (r'\bI require\b', 0.8),
        (r'\bthis is not negotiable\b', 0.9),
        (r'\bend of discussion\b', 0.9),
        (r'\bbecause I said so\b', 0.9),
        (r'\bmy decision\b', 0.75),
        (r'\bthat\'s final\b', 0.85),
        (r'\bI\'m the\b', 0.6),
        (r'\bI\'ve decided\b', 0.7),
        (r'\bwe will\b', 0.45),
        (r'\bI expect\b', 0.65),
    ],
}

_URGENCY_PATTERNS: List[Tuple[str, UrgencyLevel, float]] = [
    (r'\bASAP\b', UrgencyLevel.CRITICAL, 0.85),
    (r'\bemergency\b', UrgencyLevel.CRITICAL, 0.95),
    (r'\bcritical\b', UrgencyLevel.CRITICAL, 0.8),
    (r'\bright now\b', UrgencyLevel.HIGH, 0.8),
    (r'\bimmediately\b', UrgencyLevel.HIGH, 0.8),
    (r'\burgent(?:ly)?\b', UrgencyLevel.HIGH, 0.85),
    (r'\bneed (?:it )?by (?:today|tomorrow|tonight)\b', UrgencyLevel.HIGH, 0.75),
    (r'\bdeadline\b', UrgencyLevel.HIGH, 0.7),
    (r'\btime sensitive\b', UrgencyLevel.HIGH, 0.75),
    (r'\bsoon\b', UrgencyLevel.MEDIUM, 0.5),
    (r'\bwhen you get a chance\b', UrgencyLevel.LOW, 0.6),
    (r'\bat your convenience\b', UrgencyLevel.LOW, 0.6),
    (r'\bno rush\b', UrgencyLevel.LOW, 0.7),
    (r'\bwhenever\b', UrgencyLevel.LOW, 0.6),
    (r'\bwhenever you can\b', UrgencyLevel.LOW, 0.65),
    (r'\bnot urgent\b', UrgencyLevel.LOW, 0.8),
    (r'\btoday\b', UrgencyLevel.MEDIUM, 0.3),
    (r'\btomorrow\b', UrgencyLevel.MEDIUM, 0.25),
    (r'\bthis week\b', UrgencyLevel.MEDIUM, 0.2),
    (r'\bnext week\b', UrgencyLevel.LOW, 0.2),
]

_COMPLEX_WORDS = {
    "leverage", "facilitate", "paradigm", "synergy", "optimize", "ideate",
    "disseminate", "amalgamate", "corroborate", "juxtapose", "elucidate",
    "promulgate", "obfuscate", "obviate", "reimagine", "orchestrate",
    "operationalize", "institutionalize", "contextualize", "problematize",
    "crystallize", "actualize", "conceptualize", "strategize", "compartmentalize",
    "disambiguate", "decompose", "enumerate", "extrapolate", "interpolate",
}

_AMBIGUITY_MARKERS = [
    r'\bmaybe\b', r'\bperhaps\b', r'\bpossibly\b', r'\bmight\b', r'\bcould\b',
    r'\bseems?\b', r'\bappears?\b', r'\bsupposedly\b', r'\bapparently\b',
    r'\ballegedly\b', r'\broughly\b', r'\bapproximately\b', r'\bsomewhat\b',
]

_MAX_TEXT_LENGTH = 10000
_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: str) -> str:
    """Strip control chars, preserve newlines/tabs and <> for content."""
    if text is None:
        return ""
    return _SANITIZE_RE.sub('', str(text)).strip()


def _sanitize_strict(text: str) -> str:
    """Strip control chars and <> for identifiers."""
    if text is None:
        return ""
    return _SANITIZE_STRICT_RE.sub('', str(text)).strip()


class ThoughtSense:
    _MAX_HISTORY = 1000
    _MAX_INTENT_PATTERNS = 500
    _MAX_EMOTION_PATTERNS = 500
    _MAX_TEXT_LENGTH = 10000

    def __init__(self):
        self._history: deque = deque(maxlen=self._MAX_HISTORY)
        self._lock = threading.Lock()
        self._intent_counter: Counter = Counter()
        self._emotion_counter: Counter = Counter()
        self._total_analyses = 0

        self._intent_patterns: Dict[str, List[Tuple[str, float]]] = copy.deepcopy(_INTENT_PATTERNS)
        self._emotion_keywords: Dict[str, Dict[str, List[str]]] = copy.deepcopy(_EMOTION_KEYWORDS)
        self._subtext_patterns: Dict[str, List[Tuple[str, float]]] = copy.deepcopy(_SUBTEXT_PATTERNS)
        self._urgency_patterns: List[Tuple[str, UrgencyLevel, float]] = list(_URGENCY_PATTERNS)

        self._compiled_intent: Dict[str, List[Tuple[re.Pattern, float]]] = {}
        self._compiled_subtext: Dict[str, List[Tuple[re.Pattern, float]]] = {}
        self._compiled_urgency: List[Tuple[re.Pattern, UrgencyLevel, float]] = []
        self._compiled_ambiguity: List[re.Pattern] = []
        self._compile_patterns()

    def _compile_patterns(self):
        for intent, patterns in self._intent_patterns.items():
            self._compiled_intent[intent] = [
                (re.compile(p, re.IGNORECASE), score) for p, score in patterns
            ]
        for stype, patterns in self._subtext_patterns.items():
            self._compiled_subtext[stype] = [
                (re.compile(p, re.IGNORECASE), score) for p, score in patterns
            ]
        self._compiled_urgency = [
            (re.compile(p, re.IGNORECASE), level, score) for p, level, score in self._urgency_patterns
        ]
        self._compiled_ambiguity = [re.compile(p, re.IGNORECASE) for p in _AMBIGUITY_MARKERS]

    def detect_intent(self, text: str) -> List[IntentResult]:
        if not text or not text.strip():
            return []
        results: Dict[str, Tuple[float, List[str]]] = {}
        for intent_val, patterns in self._compiled_intent.items():
            total_score = 0.0
            matched_keywords: List[str] = []
            for pattern, base_score in patterns:
                match = pattern.search(text)
                if match:
                    total_score += base_score
                    matched_keywords.append(match.group())
            if total_score > 0:
                confidence = min(1.0, total_score)
                results[intent_val] = (confidence, matched_keywords)

        sorted_results = sorted(results.items(), key=lambda x: x[1][0], reverse=True)
        top = sorted_results[:3]
        return [
            IntentResult(intent=intent_val, confidence=conf, keywords_matched=kws)
            for intent_val, (conf, kws) in top
        ]

    def detect_emotion(self, text: str, threshold: float = 0.3) -> List[EmotionResult]:
        if not text or not text.strip():
            return []
        text_lower = text.lower()
        text_words = set(re.findall(r"[a-z][a-z'\-]*", text_lower))
        results: Dict[str, Tuple[float, List[str]]] = {}
        for emotion_val, intensity_map in self._emotion_keywords.items():
            total_intensity = 0.0
            matched_keywords: List[str] = []
            matched_set: set = set()
            for intensity_level, keywords in intensity_map.items():
                multiplier = {"mild": 0.4, "moderate": 0.7, "strong": 1.0}.get(intensity_level, 0.5)
                for kw in keywords:
                    kw_lower = kw.lower()
                    if kw_lower in matched_set:
                        continue
                    kw_words = kw_lower.split()
                    if len(kw_words) > 1:
                        if kw_lower in text_lower:
                            total_intensity += multiplier
                            matched_keywords.append(kw)
                            matched_set.add(kw_lower)
                    elif kw_lower in text_words:
                        total_intensity += multiplier
                        matched_keywords.append(kw)
                        matched_set.add(kw_lower)
            if total_intensity >= threshold:
                final_intensity = min(1.0, total_intensity)
                results[emotion_val] = (final_intensity, matched_keywords)

        sorted_results = sorted(results.items(), key=lambda x: x[1][0], reverse=True)
        return [
            EmotionResult(emotion=emotion_val, intensity=intensity, keywords_matched=kws)
            for emotion_val, (intensity, kws) in sorted_results
        ]

    def detect_subtext(self, text: str) -> List[SubtextResult]:
        if not text or not text.strip():
            return []
        results: Dict[str, Tuple[float, List[str]]] = {}
        for stype, patterns in self._compiled_subtext.items():
            total_confidence = 0.0
            evidence: List[str] = []
            for pattern, base_confidence in patterns:
                for match in pattern.finditer(text):
                    total_confidence += base_confidence
                    evidence.append(match.group())
            if total_confidence > 0:
                final_confidence = min(1.0, total_confidence)
                results[stype] = (final_confidence, evidence)

        sorted_results = sorted(results.items(), key=lambda x: x[1][0], reverse=True)
        return [
            SubtextResult(subtext_type=stype, confidence=conf, evidence=ev)
            for stype, (conf, ev) in sorted_results
        ]

    def estimate_cognitive_load(self, text: str) -> float:
        if not text or not text.strip():
            return 0.0
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0.0

        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)

        words = text.lower().split()
        if not words:
            return 0.0
        complex_count = sum(1 for w in words if w.rstrip(".,;:!?") in _COMPLEX_WORDS)
        vocab_complexity = complex_count / len(words)

        unique_stems = len(set(w[:6] for w in words if len(w) >= 6))
        topic_density = min(1.0, unique_stems / max(len(words), 1))

        ambiguity_count = sum(1 for p in self._compiled_ambiguity if p.search(text))
        ambiguity_score = min(1.0, ambiguity_count / max(len(sentences), 1))

        score = (
            avg_sentence_length / 25.0 * 30.0
            + vocab_complexity * 30.0
            + topic_density * 20.0
            + ambiguity_score * 20.0
        )
        return round(min(100.0, max(0.0, score)), 1)

    def assess_urgency(self, text: str) -> Tuple[str, float]:
        if not text or not text.strip():
            return (UrgencyLevel.LOW.value, 0.0)
        urgency_scores: Dict[str, float] = {level.value: 0.0 for level in UrgencyLevel}
        for pattern, level, base_score in self._compiled_urgency:
            if pattern.search(text):
                urgency_scores[level.value] += base_score

        level_priority = {
            UrgencyLevel.CRITICAL.value: 4,
            UrgencyLevel.HIGH.value: 3,
            UrgencyLevel.MEDIUM.value: 2,
            UrgencyLevel.LOW.value: 1,
        }
        best_level = max(urgency_scores, key=lambda k: (urgency_scores[k], level_priority.get(k, 0)))
        best_score = urgency_scores[best_level]

        if best_score == 0:
            return (UrgencyLevel.LOW.value, 0.1)

        confidence = min(1.0, best_score)
        return (best_level, round(confidence, 2))

    def analyze(self, text: str) -> ThoughtAnalysis:
        if not text or not text.strip():
            empty = ThoughtAnalysis(
                text="",
                intents=[],
                emotions=[],
                subtexts=[],
                cognitive_load=0.0,
                urgency=UrgencyLevel.LOW.value,
                urgency_confidence=0.0,
                timestamp=datetime.now(timezone.utc).isoformat(),
                summary="Empty text provided",
            )
            return empty
        text = _sanitize(text.strip())[:self._MAX_TEXT_LENGTH]
        intents = self.detect_intent(text)
        emotions = self.detect_emotion(text)
        subtexts = self.detect_subtext(text)
        cognitive_load = self.estimate_cognitive_load(text)
        urgency, urgency_conf = self.assess_urgency(text)

        summary = self._generate_summary(text, intents, emotions, subtexts, cognitive_load, urgency)

        analysis = ThoughtAnalysis(
            text=text,
            intents=intents,
            emotions=emotions,
            subtexts=subtexts,
            cognitive_load=cognitive_load,
            urgency=urgency,
            urgency_confidence=urgency_conf,
            timestamp=datetime.now(timezone.utc).isoformat(),
            summary=summary,
        )

        with self._lock:
            self._history.append(copy.deepcopy(analysis))
            self._total_analyses += 1
            for ir in intents:
                self._intent_counter[ir.intent] += 1
            for er in emotions:
                self._emotion_counter[er.emotion] += 1

        return analysis

    def analyze_batch(self, texts: List[str]) -> List[ThoughtAnalysis]:
        return [self.analyze(t) for t in texts]

    def _generate_summary(
        self,
        text: str,
        intents: List[IntentResult],
        emotions: List[EmotionResult],
        subtexts: List[SubtextResult],
        cognitive_load: float,
        urgency: str,
    ) -> str:
        parts = []
        if intents:
            top_intent = intents[0]
            parts.append(f"Primary intent: {top_intent.intent} ({top_intent.confidence:.0%})")
        if emotions:
            top_emotion = emotions[0]
            parts.append(f"Dominant emotion: {top_emotion.emotion} (intensity {top_emotion.intensity:.1f})")
        if subtexts:
            top_subtext = subtexts[0]
            parts.append(f"Subtext: {top_subtext.subtext_type} ({top_subtext.confidence:.0%})")
        if cognitive_load > 60:
            parts.append(f"High cognitive load ({cognitive_load:.0f}/100)")
        elif cognitive_load > 30:
            parts.append(f"Moderate cognitive load ({cognitive_load:.0f}/100)")
        if urgency in (UrgencyLevel.HIGH.value, UrgencyLevel.CRITICAL.value):
            parts.append(f"Urgency: {urgency}")
        if not parts:
            parts.append("Neutral communication with no strong signals detected")
        return ". ".join(parts) + "."

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "total_analyses": self._total_analyses,
                "history_size": len(self._history),
                "top_intents": self._intent_counter.most_common(5),
                "top_emotions": self._emotion_counter.most_common(5),
            }

    def clear_history(self):
        with self._lock:
            self._history.clear()
            self._intent_counter.clear()
            self._emotion_counter.clear()
            self._total_analyses = 0