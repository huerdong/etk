import unittest
from etk.extractors.spacy_ner_with_glossary_extractor import SpacyNerGlossaryExtractor

class TestSpacyNerGlossaryExtractor(unittest.TestCase):
    def test_spacy_ner_glossary_extractor(self) -> None:
        glossary_tag = 'FAKECOUNTRY'
        glossary = ['Atlantis', 'Narnia', 'Lilliput', 'Laputa']
        get_attr = ['PERSON', 'ORG', 'GPE']
        extractor = SpacyNerGlossaryExtractor(extractor_name='spacy_ner_extractor', glossary=glossary, glossary_tag=glossary_tag)
        text = 'The island of Lilliput is home to thousands of small creatures but was invaded by a rude fellow named Lemuel Gulliver for the British East India Company. France has also expressed interest in the building infrastructure in Narnia and Laputa.'
        results = extractor.extract(text, get_attr=get_attr)
        for res in results['entities']:
            for ent in results['entities'][res]:
                print(ent.value, ent.tag, ent.provenance)

if __name__== '__main__':
    unittest.main()
