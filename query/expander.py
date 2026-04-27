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
    "sme": "SME small medium enterprise financing Pakistan",
    "agriculture": "agriculture loan financing Pakistan Zarai Taraqiati Bank",
    "ztbl": "ZTBL Zarai Taraqiati Bank Limited agriculture loan Pakistan",
}

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


def expand_query(query: str, language: str = "english") -> str:
    normalized = query.strip().lower()
    if language == "urdu":
        expansion_map = URDU_EXPANSION_MAP
        for key, expansion in expansion_map.items():
            if key in query:
                return f"{query} {expansion}"
        return query
    if language == "roman_urdu":
        expansion_map = ROMAN_URDU_EXPANSION_MAP
    else:
        expansion_map = EXPANSION_MAP
    if normalized in expansion_map:
        return f"{query} {expansion_map[normalized]}"
    for key, expansion in expansion_map.items():
        if key in normalized and len(normalized.split()) <= 2:
            return f"{query} {expansion}"
    return query