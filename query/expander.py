EXPANSION_MAP: dict[str, str] = {
    "zakat": "zakat calculation Pakistan nisab 2.5% gold silver",
    "riba": "riba interest Islamic finance prohibition haram",
    "halal": "halal investment Pakistan Islamic finance permissible",
    "haram": "haram investment prohibited Islamic finance",
    "sukuk": "sukuk Islamic bond Pakistan investment",
    "murabaha": "murabaha Islamic financing cost plus sale",
    "musharakah": "musharakah partnership Islamic finance profit loss",
    "mudarabah": "mudarabah profit sharing Islamic finance",
    "ijarah": "ijarah Islamic leasing finance",
    "takaful": "takaful Islamic insurance Pakistan",
    "nisab": "nisab zakat threshold gold silver Pakistan",
    "ushr": "ushr agricultural zakat 10 percent crop",
    "tax": "income tax Pakistan FBR filing return",
    "income tax": "income tax Pakistan FBR slab rate filing",
    "fbr": "FBR Federal Board of Revenue Pakistan tax filing",
    "secp": "SECP Securities Exchange Commission Pakistan regulation",
    "sbp": "SBP State Bank of Pakistan monetary policy regulation",
    "inflation": "inflation Pakistan CPI consumer price index rate",
    "forex": "foreign exchange rate Pakistan dollar rupee",
    "remittance": "remittance Pakistan overseas worker transfer",
    "loan": "loan interest rate Pakistan bank financing",
    "mortgage": "mortgage home loan Pakistan bank financing",
    "pension": "pension retirement fund Pakistan EOBI",
    "eobi": "EOBI employees old age benefits institution Pakistan pension",
    "profit": "profit rate saving account Pakistan bank",
    "saving": "saving account profit rate Pakistan bank",
    "investment": "investment options Pakistan mutual fund stock bond",
    "mutual fund": "mutual fund Pakistan AMC returns NAV",
    "stock": "stock share Pakistan PSX KSE investment",
    "psx": "PSX Pakistan Stock Exchange shares trading",
    "dividend": "dividend income tax Pakistan company shares",
    "capital gain": "capital gain tax Pakistan shares property",
    "property tax": "property tax Pakistan FBR withholding immovable",
    "withholding": "withholding tax Pakistan FBR deduction filer",
    "filer": "filer tax return Pakistan FBR active taxpayer",
    "nonfiler": "non filer withholding tax Pakistan higher rate",
    "non filer": "non filer withholding tax Pakistan higher rate",
    "prize bond": "prize bond Pakistan national savings halal haram",
    "national savings": "national savings Pakistan certificates profit DSC SSC",
    "insurance": "insurance Pakistan conventional takaful policy",
    "bancassurance": "bancassurance bank insurance Pakistan policy",
    "microfinance": "microfinance Pakistan small loan poor rural",
    "freelance": "freelance freelancing Pakistan income tax FBR export services registration benefits",
    "freelancing": "freelancing Pakistan freelance income Upwork Fiverr tax FBR registration benefits",
    "sme": "SME small medium enterprise financing Pakistan",
    "agriculture": "agriculture loan financing Pakistan Zarai Taraqiati Bank",
    "ztbl": "ZTBL Zarai Taraqiati Bank Limited agriculture loan Pakistan",
}

# Expand even when the query has many tokens (substring match); avoids missing long Roman Urdu questions.
ALWAYS_EXPAND_SUBSTR_KEYS = frozenset({"freelance", "freelancing", "investment"})


ROMAN_URDU_EXPANSION_MAP: dict[str, str] = {
    "zakat": "zakat calculation nisab Pakistan 2.5 gold silver",
    "riba": "riba sood interest Islamic prohibition haram",
    "halal": "halal investment Islamic finance Pakistan jaiz",
    "haram": "haram investment prohibited Islamic finance najaiz",
    "loan": "loan qarz interest rate bank Pakistan",
    "tax": "tax Pakistan FBR income filing return",
    "saving": "saving account profit rate bank Pakistan",
    "nifaq": "nifaq double dealing financial fraud",
    "qarz": "qarz loan interest riba Islamic finance",
    "sood": "sood riba interest prohibition Islamic banking",
    "munafa": "munafa profit rate bank saving account",
    "freelance": "freelance freelancing Pakistan kamai tax FBR registration faida nuqsan",
    "freelancing": "freelancing Pakistan freelance income kamai tax FBR registration",
}

URDU_EXPANSION_MAP: dict[str, str] = {
    "زکات": "زکات حساب نصاب پاکستان سونا چاندی",
    "سود": "سود ربا حرام اسلامی بینکاری",
    "حلال": "حلال سرمایہ کاری اسلامی مالیات",
    "حرام": "حرام سرمایہ کاری ممنوع اسلامی مالیات",
    "ٹیکس": "ٹیکس پاکستان ایف بی آر آمدنی",
    "قرض": "قرض لون بینک پاکستان مالیات",
    "منافع": "منافع شرح بینک بچت کھاتہ",
    "بیمہ": "بیمہ تکافل پاکستان پالیسی",
    "سرمایہ": "سرمایہ کاری میوچوئل فنڈ پاکستان",
}

ROMAN_URDU_TO_ENGLISH_GLOSSARY: dict[str, str] = {
    "sarkari": "government",
    "hukoomati": "government",
    "hukumati": "government",
    "hukumat": "government",
    "hukoomat": "government",
    "mulazim": "employee",
    "mulazimeen": "employees",
    "naukri": "job employment",
    "naukar": "employee worker",
    "tankhah": "salary",
    "tankhwah": "salary",
    "tankha": "salary",
    "salaried": "salaried employee",
    "aamdani": "income",
    "aamadni": "income",
    "amdani": "income",
    "kamai": "income earnings",
    "kharcha": "expense expenditure",
    "kharchay": "expenses",
    "bachat": "savings",
    "khaata": "account",
    "khata": "account",
    "kholna": "open opening",
    "khulwana": "open opening",
    "qist": "installment",
    "kist": "installment",
    "kistein": "installments",
    "qarz": "loan debt",
    "qarza": "loan",
    "udhar": "loan credit",
    "sood": "interest",
    "munafa": "profit return",
    "munafe": "profits returns",
    "moaaf": "exempt exemption",
    "moaafi": "exemption",
    "mafi": "exemption pardon",
    "chhut": "exemption relief",
    "chhoot": "exemption relief",
    "ada": "pay payment",
    "wasool": "collect collection",
    "wasooli": "collection recovery",
    "ghar": "house property",
    "makaan": "house property",
    "zameen": "land property",
    "jageer": "property estate",
    "karobar": "business trade",
    "kaaribar": "business",
    "tijarat": "trade business commercial",
    "tajir": "trader merchant",
    "dukan": "shop store",
    "dukandar": "shopkeeper merchant",
    "saalana": "annual yearly",
    "maheena": "monthly month",
    "haftawar": "weekly",
    "rozana": "daily",
    "qanoon": "law statute",
    "qanooni": "legal statutory",
    "dafa": "section clause",
    "shoba": "section department",
    "ehtiyat": "reserve provision",
    "daulat": "wealth assets",
    "amwal": "assets wealth property",
    "maal": "wealth assets goods",
    "rakam": "amount sum",
    "raqam": "amount sum",
    "shahri": "citizen resident",
    "shehri": "citizen resident",
    "mulki": "domestic resident",
    "pardesi": "foreign overseas",
    "bidesi": "foreign overseas",
    "videsi": "foreign overseas",
    "wasiyat": "will inheritance",
    "wirsa": "inheritance heir",
    "warasat": "inheritance succession",
    "warasati": "inheritance",
    "shadi": "marriage",
    "talaq": "divorce",
    "wakeel": "lawyer attorney",
    "darkhwast": "application",
    "munsif": "judge magistrate",
    "adalat": "court tribunal",
    "muqadma": "case lawsuit",
    "shikayat": "complaint",
}

ROMAN_URDU_TO_ENGLISH_PHRASES: dict[str, str] = {
    "sarkari mulazim": "government employee",
    "sarkari mulazimeen": "government employees",
    "sarkari naukri": "government job",
    "tax slab": "tax slab rate bracket",
    "tax chhut": "tax exemption",
    "tax moaafi": "tax exemption",
    "income tax": "income tax FBR slab rate filing",
    "qarz lena": "borrow loan",
    "qarz dena": "lend loan",
    "khaata kholna": "open account",
    "bank account kholna": "open bank account",
}

URDU_TO_ENGLISH_GLOSSARY: dict[str, str] = {
    "سرکاری": "government",
    "ملازم": "employee",
    "ملازمین": "employees",
    "تنخواہ": "salary",
    "آمدن": "income",
    "آمدنی": "income",
    "بچت": "savings",
    "قسط": "installment",
    "قرض": "loan debt",
    "سود": "interest",
    "منافع": "profit",
    "چھوٹ": "exemption",
    "معافی": "exemption",
    "ادا": "pay",
    "وصول": "collect",
    "گھر": "house property",
    "مکان": "house property",
    "زمین": "land property",
    "کاروبار": "business trade",
    "تجارت": "trade business",
    "دکان": "shop",
    "دکاندار": "shopkeeper",
    "ٹیکس": "tax",
    "حکومت": "government",
    "حکومتی": "government",
    "قانون": "law",
    "دفعہ": "section",
    "املاک": "assets property",
    "وراثت": "inheritance",
    "ادالت": "court",
    "عدالت": "court",
}


def _apply_glossary(query: str, glossary: dict[str, str], phrases: dict[str, str] | None = None) -> str:
    """Append English equivalents of Roman Urdu / Urdu tokens so BM25 can match English PDFs."""
    additions: list[str] = []
    lowered = query.lower()

    if phrases:
        for phrase, english in phrases.items():
            if phrase in lowered:
                additions.append(english)

    tokens = lowered.split()
    seen: set[str] = set()
    for tok in tokens:
        clean = tok.strip(".,;:!?\"'()[]{}")
        if clean and clean in glossary and clean not in seen:
            additions.append(glossary[clean])
            seen.add(clean)

    if not additions:
        return query
    return f"{query} {' '.join(additions)}"


import logging
log = logging.getLogger(__name__)


def expand_query(query: str, language: str = "english") -> str:
    normalized = query.strip().lower()
    original = query
    if language == "urdu":
        result = query
        for key, expansion in URDU_EXPANSION_MAP.items():
            if key in query:
                result = f"{result} {expansion}"
                break
        result = _apply_glossary(result, URDU_TO_ENGLISH_GLOSSARY)
        if result != original:
            log.info("expand_query[urdu]: '%s' -> '%s'", original, result)
        return result

    if language == "roman_urdu":
        expansion_map = ROMAN_URDU_EXPANSION_MAP
    else:
        expansion_map = EXPANSION_MAP

    expanded = query
    if normalized in expansion_map:
        expanded = f"{query} {expansion_map[normalized]}"
    else:
        for key, expansion in expansion_map.items():
            if key not in normalized:
                continue
            if len(normalized.split()) <= 2 or key in ALWAYS_EXPAND_SUBSTR_KEYS:
                expanded = f"{query} {expansion}"
                break

    if language == "roman_urdu":
        expanded = _apply_glossary(expanded, ROMAN_URDU_TO_ENGLISH_GLOSSARY, ROMAN_URDU_TO_ENGLISH_PHRASES)

    if expanded != original:
        log.info("expand_query[%s]: '%s' -> '%s'", language, original, expanded)
    else:
        log.debug("expand_query[%s]: no expansion for '%s'", language, original)

    return expanded