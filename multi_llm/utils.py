def parse_final_answer(text):
    answer_start = "Answer (Yes, No or Neither):"
    end_marker = "<end_of_turn>"
    
    if answer_start in text:
        text = text.split(answer_start)[-1].strip()
    if end_marker in text:
        text = text.split(end_marker)[0].strip()
    return text

def parse_response(text, prefix):
    end_marker = "<end_of_turn>"
    if prefix in text:
        text = text.split(prefix)[-1].strip()
    if end_marker in text:
        text = text.split(end_marker)[0].strip()
    return text
    
country_capitalized_mapping = {
        'malaysia': 'Malaysia',
        'lebanon': 'Lebanon',
        'colombia': 'Colombia',
        'romania': 'Romania',
        'south_africa': 'South Africa',
        'myanmar': 'Myanmar',
        'saudi_arabia': 'Saudi Arabia',
        'samoa': 'Samoa',
        'sudan': 'Sudan',
        'united_kingdom': 'United Kingdom',
        'australia': 'Australia',
        'taiwan': 'Taiwan',
        'thailand': 'Thailand',
        'pakistan': 'Pakistan',
        'philippines': 'Philippines',
        'sri_lanka': 'Sri Lanka',
        'north_macedonia': 'North Macedonia',
        'germany': 'Germany',
        'cambodia': 'Cambodia',
        'türkiye': 'Türkiye',
        'zimbabwe': 'Zimbabwe',
        'fiji': 'Fiji',
        'hungary': 'Hungary',
        'mexico': 'Mexico',
        'brazil': 'Brazil',
        'palestinian_territories': 'Palestinian Territories',
        'iraq': 'Iraq',
        'poland': 'Poland',
        'sweden': 'Sweden',
        'ukraine': 'Ukraine',
        'bangladesh': 'Bangladesh',
        'new_zealand': 'New Zealand',
        'france': 'France',
        'ethiopia': 'Ethiopia',
        'somalia': 'Somalia',
        'portugal': 'Portugal',
        'croatia': 'Croatia',
        'united_states_of_america': 'United States of America',
        'egypt': 'Egypt',
        'china': 'China',
        'mauritius': 'Mauritius',
        'cyprus': 'Cyprus',
        'greece': 'Greece',
        'italy': 'Italy',
        'south_sudan': 'South Sudan',
        'venezuela': 'Venezuela',
        'iran': 'Iran',
        'hong_kong': 'Hong Kong',
        'timor-leste': 'Timor-Leste',
        'austria': 'Austria',
        'india': 'India',
        'israel': 'Israel',
        'nepal': 'Nepal',
        'bosnia_and_herzegovina': 'Bosnia and Herzegovina',
        'malta': 'Malta',
        'ireland': 'Ireland',
        'peru': 'Peru',
        'vietnam': 'Vietnam',
        'singapore': 'Singapore',
        'papua_new_guinea': 'Papua New Guinea',
        'japan': 'Japan',
        'indonesia': 'Indonesia',
        'netherlands': 'Netherlands',
        'kenya': 'Kenya',
        'argentina': 'Argentina',
        'chile': 'Chile',
        'serbia': 'Serbia',
        'afghanistan': 'Afghanistan',
        'canada': 'Canada',
        'spain': 'Spain',
        'syria': 'Syria',
        'laos': 'Laos',
        'tonga': 'Tonga',
        'russia': 'Russia',
        'south_korea': 'South Korea'
    }


country_iso_mapping = {
    'malaysia': 'ms', 'lebanon': 'ar', 'colombia': 'es', 'romania': 'ro', 'south_africa': 'af', 'myanmar': 'my', 
    'saudi_arabia': 'ar', 'samoa': 'sm', 'sudan': 'ar', 'united_kingdom': 'en', 'australia': 'en', 'taiwan': 'zh-TW', 
    'thailand': 'th', 'pakistan': 'ur', 'philippines': 'tl', 'sri_lanka': 'si', 'north_macedonia': 'mk', 
    'germany': 'de', 'cambodia': 'km', 'türkiye': 'tr', 'zimbabwe': 'en', 'fiji': 'fj', 'hungary': 'hu', 
    'mexico': 'es', 'brazil': 'pt', 'palestinian_territories': 'ar', 'iraq': 'ar', 'poland': 'pl', 'sweden': 'sv', 
    'ukraine': 'uk', 'bangladesh': 'bn', 'new_zealand': 'en', 'france': 'fr', 'ethiopia': 'am', 'somalia': 'so', 
    'portugal': 'pt', 'croatia': 'hr', 'united_states_of_america': 'en', 'egypt': 'ar', 'china': 'zh-CN', 
    'mauritius': 'fr', 'cyprus': 'el', 'greece': 'el', 'italy': 'it', 'south_sudan': 'en', 'venezuela': 'es', 
    'iran': 'fa', 'hong_kong': 'zh-TW', 'timor-leste': 'pt', 'austria': 'de', 'india': 'hi', 'israel': 'iw', 
    'nepal': 'ne', 'bosnia_and_herzegovina': 'bs', 'malta': 'mt', 'ireland': 'ga', 'peru': 'es', 'vietnam': 'vi', 
    'singapore': 'en', 'papua_new_guinea': 'en', 'japan': 'ja', 'indonesia': 'id', 'netherlands': 'nl', 
    'kenya': 'sw', 'argentina': 'es', 'chile': 'es', 'serbia': 'sr', 'afghanistan': 'ps', 'canada': 'en', 
    'spain': 'es', 'syria': 'ar', 'laos': 'lo', 'tonga': 'to', 'russia': 'ru', 'south_korea': 'ko'
}


country_native_mapping = {
    'malaysia': 'Malaysia', 'lebanon': 'لبنان', 'colombia': 'Colombia', 'romania': 'România', 'south_africa': 'Suid-Afrika', 'myanmar': 'မြန်မာ', 'saudi_arabia': 'المملكة العربية السعودية', 'samoa': 'Samoa', 'sudan': 'السودان', 'united_kingdom': 'United Kingdom', 'australia': 'Australia', 'taiwan': '台灣', 'thailand': 'ประเทศไทย', 'pakistan': 'پاکستان', 'philippines': 'Pilipinas', 'sri_lanka': 'ශ්\u200dරී ලංකාව', 'north_macedonia': 'Северна Македонија', 'germany': 'Deutschland', 'cambodia': 'កម្ពុជា។', 'türkiye': 'Türkiye', 'zimbabwe': 'Zimbabwe', 'hungary': 'Magyarország', 'mexico': 'México', 'brazil': 'Brasil', 'palestinian_territories': 'الأراضي الفلسطينية', 'iraq': 'العراق', 'poland': 'Polska', 'sweden': 'Sverige', 'ukraine': 'Україна', 'bangladesh': 'বাংলাদেশ', 'new_zealand': 'New Zealand', 'france': 'France', 'ethiopia': 'ኢትዮጵያ', 'somalia': 'Soomaaliya', 'portugal': 'Portugal', 'croatia': 'Hrvatska', 'united_states_of_america': 'United States of America', 'egypt': 'مصر', 'china': '中国', 'mauritius': 'Ile Maurice', 'cyprus': 'Κύπρος', 'greece': 'Ελλάδα', 'italy': 'Italia', 'south_sudan': 'South Sudan', 'venezuela': 'Venezuela', 'iran': 'ایران', 'hong_kong': '香港', 'timor-leste': 'Timor-Leste', 'austria': 'Österreich', 'india': 'भारत', 'israel': 'יִשְׂרָאֵל', 'nepal': 'नेपाल', 'bosnia_and_herzegovina': 'Bosna i Hercegovina', 'malta': 'Malta', 'ireland': 'Éire', 'peru': 'Perú', 'vietnam': 'Việt Nam', 'singapore': 'Singapore', 'papua_new_guinea': 'Papua New Guinea', 'japan': '日本', 'indonesia': 'Indonesia', 'netherlands': 'Nederland', 'kenya': 'Kenya', 'argentina': 'Argentina', 'chile': 'Chile', 'serbia': 'Србија', 'afghanistan': 'افغانستان', 'canada': 'Canada', 'spain': 'España', 'syria': 'سوريا', 'laos': 'ປະເທດລາວ', 'russia': 'Россия', 'south_korea': '대한민국'
}