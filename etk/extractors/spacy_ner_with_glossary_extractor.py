import spacy
from etk.extractor import Extractor, InputType
from etk.extraction import Extraction
from typing import List
import re

class SpacyNerGlossaryExtractor(Extractor):
    """
    **Description**
        This extractor takes a list of spaCy NER tag as reference, and extract
        the tag matched substring from the input text. It also takes a glossary 
        to extract glossary-defined entities

    Examples:
        ::
            
            glossary_tag = 'Countries'
            glossary = ['Spain', 'China', 'Narnia', 'Lilliput']
            get_attr = ['PERSON', 'ORG', 'GPE']
            spacy_ner_extractor = SpacyNerExtractor(glossary=glossary, glossary_name=glossary_tag)
            spacy_ner_extractor.extract(text=text, get_attr=get_attr)

    """
    def __init__(self, extractor_name: str, glossary: List[str], glossary_tag: str, nlp=spacy.load('en_core_web_sm')):
        Extractor.__init__(self, input_type=InputType.TEXT,
                           category="built_in_extractor",
                           name=extractor_name)
        self.__nlp = nlp
        self.glossary_tag = glossary_tag
        self.glossary = glossary

    # all_attrs = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE',
    #              'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']
    def extract(self, text: str, get_attr=['PERSON', 'ORG', 'GPE']):
        """
        Args:
            text (str): the text to extract from.
            get_attr (List[str]): The spaCy NER attributes we're interested in.

        Returns:
            dict: dictionary with entities which has all extracted tokens 

        """
        results={}

        entities = {}
        entities[self.glossary_tag] = []
        for label in get_attr:
            entities[label] = []

        doc = self.__nlp(text)

        # Naively extract case sensitive exact matches (Needs to be refined later TODO)
        for term in self.glossary:
            matcher = r"\b" + re.escape(term) + r"\b"
            length = len(term)
            matched = [m for m in re.finditer(matcher, text)]
            entities[self.glossary_tag].extend([Extraction(extractor_name=self.name,
                    start_char=m.start(),
                    end_char=(m.start() + length),
                    value=term,
                    tag=self.glossary_tag)
                    for m in matched
                ])

        for ent in doc.ents:
            if ent.label_ in get_attr:
                entities[ent.label_].append(Extraction(extractor_name=self.name,
                                            start_char=int(ent.start_char),
                                            end_char=int(ent.end_char),
                                            value=ent.text,
                                            tag=ent.label_))

        return {"entities": entities}
