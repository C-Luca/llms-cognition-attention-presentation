"""
========================================================================================
Module: Results - real WMT 2014 sample sentences
Author: Gusev Roman (Results Interpreter & Challenger)
========================================================================================
Real sentence pairs copied (whitespace normalised, nothing else changed) from the public
Hugging Face dataset `wmt/wmt14` (configs `de-en` and `fr-en`). They are hard-coded so the
presentation works offline. `row` is the position of the pair inside its split.

TRAIN_*: WMT 2014 training data (these particular rows are Europarl, i.e. European Parliament debates).
DEV    : newstest2013, the development set used for Table 3 (same English in the de-en and fr-en sets).
TEST   : newstest2014, the test set used for Table 2. The same English sentence appears in
         the German and the French test set, so one source line has two references.
========================================================================================
"""

TRAIN_DE = [
    {
        "row": 1000003,
        "en": "Let us give due consideration to these results.",
        "de": "Wir sollten über diese Ergebnisse nachdenken."
    },
    {
        "row": 1000012,
        "en": "If so, why have these figures not yet been made public?",
        "de": "Wenn ja, warum sind diese Zahlen bislang nicht bekannt geworden?"
    },
    {
        "row": 1000014,
        "en": "Where are these applications examined?",
        "de": "Wo werden diese Anträge untersucht?"
    },
    {
        "row": 1000024,
        "en": "Frontex is exempt from all parliamentary control.",
        "de": "Es gibt bezüglich Frontex keine parlamentarische Kontrolle."
    }
]

TRAIN_FR = [
    {
        "row": 2000002,
        "en": "I have already described the current state of affairs in my reply.",
        "fr": "J'ai déjà décrit dans ma réponse l'état actuel des choses."
    },
    {
        "row": 2000011,
        "en": "Madam President-in-Office of the Council, forgive me.",
        "fr": "Madame la Présidente en exercice du Conseil, excusez-moi !"
    },
    {
        "row": 2000023,
        "en": "Thank you very much, Mr Korakas.",
        "fr": "Merci beaucoup, Monsieur Korakas."
    },
    {
        "row": 2000029,
        "en": "That is a different matter from reconstruction aid proper.",
        "fr": "C'est autre chose que l'aide à la reconstruction proprement dite."
    }
]

DEV = [
    {
        "row": 0,
        "en": "A Republican strategy to counter the re-election of Obama",
        "de": "Eine republikanische Strategie, um der Wiederwahl von Obama entgegenzutreten",
        "fr": "Une stratégie républicaine pour contrer la réélection d'Obama"
    },
    {
        "row": 13,
        "en": "Indiana was the first State to impose such a requirement.",
        "de": "Indiana war der erste Bundesstaat, der eine derartige Forderung erhoben hat.",
        "fr": "L'Indiana fut le premier État à poser une telle exigence."
    },
    {
        "row": 33,
        "en": "These restrictions are not without consequence.",
        "de": "Diese Einschränkungen sind nicht ohne Folgen.",
        "fr": "Ces restrictions ne sont pas sans conséquence."
    },
    {
        "row": 52,
        "en": "Prostate cancer screening: take the test or not?",
        "de": "Prostatakrebsfrüherkennung: sollte der Test gemacht werden oder nicht?",
        "fr": "Dépistage du cancer de la prostate: passer le test ou non?"
    }
]

TEST = [
    {
        "en": "A black box in your car?",
        "de": "Eine Blackbox im Auto?",
        "fr": "Une boîte noire dans votre voiture ?",
        "row_de": 39,
        "row_fr": 3
    },
    {
        "en": "The tea party is aghast.",
        "de": "Die Tea Party ist entsetzt.",
        "fr": "Le Tea Party est atterré.",
        "row_de": 44,
        "row_fr": 8
    },
    {
        "en": "The technology is there to do it.",
        "de": "Die Technologie dafür ist da.",
        "fr": "La technologie est là pour le faire.",
        "row_de": 52,
        "row_fr": 16
    },
    {
        "en": "Nevada has already completed a pilot.",
        "de": "Nevada hat bereits ein Pilotprojekt abgeschlossen.",
        "fr": "Le Nevada a déjà mené à terme un projet pilote.",
        "row_de": 74,
        "row_fr": 38
    }
]
