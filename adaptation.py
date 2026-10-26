"""Conservative lexical learning. Suggestions are evidence, never proof of intent."""
import difflib
import re

WORD = re.compile(r"\b[\w]+(?:['’-][\w]+)*\b", re.UNICODE)


def words(text):
    return WORD.findall(text)


def replacements(original, corrected):
    before, after = words(original), words(corrected)
    if not before or not after:
        return []
    matcher = difflib.SequenceMatcher(None, [w.casefold() for w in before],
                                    [w.casefold() for w in after], autojunk=False)
    # Wholesale rewrites are useful labels but unsafe lexical evidence.
    if matcher.ratio() < .45 and max(len(before), len(after)) > 4:
        return []
    result = []
    for op, a, b, c, d in matcher.get_opcodes():
        if op == 'replace' and 0 < b-a <= 4 and 0 < d-c <= 4:
            source, target = ' '.join(before[a:b]), ' '.join(after[c:d])
            if len(source) <= 100 and len(target) <= 100:
                result.append((source, target))
    return result


def entity_like(phrase):
    # Conservative heuristic, not a claim to perform named-entity recognition.
    return any(any(c.isupper() for c in word[1:]) or word.isupper() or
               (word[:1].isupper() and len(word) >= 3) or any(c.isdigit() for c in word)
               for word in phrase.split())


def apply_rules(text, rules):
    """Single pass, whole-token substitutions. Never cascade one rule into another."""
    usable = {r['source'].casefold(): r for r in rules if r.get('enabled')}
    if not usable:
        return text, []
    alternatives = []
    for phrase in sorted(usable, key=lambda x: (-len(x), x)):
        alternatives.append(r'\s+'.join(re.escape(w) for w in phrase.split()))
    pattern = re.compile(r'(?<!\w)(?:' + '|'.join(alternatives) + r')(?!\w)', re.I)
    applied = []
    def replace(match):
        key = ' '.join(match.group().split()).casefold()
        rule = usable[key]
        applied.append({'source': match.group(), 'target': rule['target'], 'rule_id': rule['id']})
        return rule['target']
    return pattern.sub(replace, text), applied
