from dataclasses import dataclass, field
from typing import Dict, List, Any

import ginza
import spacy
from spacy import Language
from spacy.tokens.token import Token
from spacy.tokens.doc import Doc

nlp: Language = spacy.load("ja_ginza")


def load_nlp_model(model_name: str):
    nlp = spacy.load(model_name)


@dataclass
class Author:
    name: str = field(default="")

    def to_dict(self):
        dst = dict(vars(self).items())
        return dst

    @staticmethod
    def from_dict(src: Dict[str, str]) -> "Author":
        return Author(**src)


@dataclass
class BibInfo:
    id: int = field(default=0)
    title: str = field(default="")
    title_en: str = field(default="")
    authors: List[Author] = field(default_factory=list)
    translators: List[Author] = field(default_factory=list)
    publisher: str = field(default="")
    publication_date: str = field(default="")

    def to_dict(self):
        dst = dict(vars(self).items())
        dst["authors"] = [sec.to_dict() for sec in self.authors]
        dst["translators"] = [sec.to_dict() for sec in self.translators]
        return dst

    @staticmethod
    def from_dict(src: Dict[str, Any]) -> "BibInfo":
        for key in ["authors", "translators"]:
            src[key] = [Author.from_dict(d) for d in src.get(key, [])]
        return BibInfo(**src)


@dataclass
class TocSection:
    index: int = field(default=0)
    title: str = field(default="")

    def to_dict(self):
        dst = dict(vars(self).items())
        return dst

    @staticmethod
    def from_dict(src: Dict[str, str]) -> "TocSection":
        return TocSection(**src)


@dataclass
class TocChapter:
    index: int = field(default=0)
    title: str = field(default="")
    sections: List[TocSection] = field(default_factory=list)

    def to_dict(self):
        dst = dict(vars(self).items())
        dst["sections"] = [sec.to_dict() for sec in self.sections]
        return dst

    @staticmethod
    def from_dict(src: Dict[str, Any]) -> "TocChapter":
        for key in ["sections"]:
            src[key] = [TocSection.from_dict(d) for d in src.get(key, [])]
        return TocChapter(**src)


@dataclass
class Toc:
    chapters: List[TocChapter] = field(default_factory=list)

    def to_dict(self):
        dst = dict(vars(self).items())
        dst["chapters"] = [sec.to_dict() for sec in self.chapters]
        return dst

    @staticmethod
    def from_dict(src: Dict[str, Any]) -> "Toc":
        for key in ["chapters"]:
            src[key] = [TocChapter.from_dict(d) for d in src.get(key, [])]
        return Toc(**src)


@dataclass(frozen=True)
class VocabPartOfSpeech:
    word: str = field(default="")
    lemma: str = field(default="")
    pos: str = field(default="")
    detail_1: str = field(default="")
    detail_2: str = field(default="")

    @staticmethod
    def from_token(token: Token) -> "VocabPartOfSpeech":
        tags: List[str] = token.tag_.split("-")
        if not tags:
            tags = ["", ""]
        elif len(tags) == 1:
            tags.append("")
        return VocabPartOfSpeech(
            word=token.text,
            lemma=token.lemma_,
            pos=token.pos_,
            detail_1=tags[0],
            detail_2=tags[1]
        )

    def to_dict(self):
        dst = dict(vars(self).items())
        return dst

    @staticmethod
    def from_dict(src: Dict[str, str]) -> "VocabPartOfSpeech":
        return VocabPartOfSpeech(**src)


@dataclass
class VocabFrequencyParagraph:
    index: int = field(default=0)
    count: int = field(default=0)

    def to_dict(self):
        dst = dict(vars(self).items())
        return dst

    @staticmethod
    def from_dict(src: Dict[str, str]) -> "VocabFrequencyParagraph":
        return VocabFrequencyParagraph(**src)


@dataclass
class VocabFrequencySection:
    index: int = field(default=0)
    count: int = field(default=0)
    paragraph: List[VocabFrequencyParagraph] = field(default_factory=list)

    def to_dict(self):
        dst = dict(vars(self).items())
        dst["paragraph"] = [sec.to_dict() for sec in self.paragraph]
        return dst

    @staticmethod
    def from_dict(src: Dict[str, Any]) -> "VocabFrequencySection":
        for key in ["paragraph"]:
            src[key] = [VocabFrequencyParagraph.from_dict(
                d) for d in src.get(key, [])]
        return VocabFrequencySection(**src)


@dataclass
class VocabFrequencyChapter:
    index: int = field(default=0)
    count: int = field(default=0)
    sections: List[VocabFrequencySection] = field(default_factory=list)

    def to_dict(self):
        dst = dict(vars(self).items())
        dst["sections"] = [sec.to_dict() for sec in self.sections]
        return dst

    @staticmethod
    def from_dict(src: Dict[str, Any]) -> "VocabFrequencyChapter":
        for key in ["sections"]:
            src[key] = [VocabFrequencySection.from_dict(
                d) for d in src.get(key, [])]
        return VocabFrequencyChapter(**src)


@dataclass
class VocabFrequency:
    count: int
    chapters: List[VocabFrequencyChapter] = field(default_factory=list)

    def to_dict(self):
        dst = dict(vars(self).items())
        dst["chapters"] = [sec.to_dict() for sec in self.chapters]
        return dst

    @staticmethod
    def from_dict(src: Dict[str, Any]) -> "VocabFrequency":
        for key in ["chapters"]:
            src[key] = [VocabFrequencyChapter.from_dict(
                d) for d in src.get(key, [])]
        return VocabFrequency(**src)


@dataclass(frozen=True)
class VocabWordPosition:
    chapter_id: int = field(default=0)
    section_id: int = field(default=0)
    paragraph_id: int = field(default=0)
    sentence_id: int = field(default=0)
    position_id: int = field(default=0)

    def to_dict(self):
        dst = dict(vars(self).items())
        return dst

    @staticmethod
    def from_dict(src: Dict[str, str]) -> "VocabWordPosition":
        return VocabWordPosition(**src)


@dataclass
class VocabWord:
    value: str = field(default="")
    part_of_speech: VocabPartOfSpeech = field(
        default_factory=VocabPartOfSpeech)
    frequency: VocabFrequency = field(default_factory=VocabFrequency)
    positions: List[VocabWordPosition] = field(default_factory=list)

    def to_dict(self):
        dst = dict(vars(self).items())
        dst["part_of_speech"] = self.part_of_speech.to_dict()
        dst["frequency"] = self.frequency.to_dict()
        dst["positions"] = [sec.to_dict() for sec in self.positions]
        return dst

    @staticmethod
    def from_dict(src: Dict[str, Any]) -> "VocabWord":
        src["part_of_speech"] = VocabPartOfSpeech.from_dict(
            src.get("part_of_speech", {}))
        src["frequency"] = VocabFrequency.from_dict(src.get("frequency", {}))
        src["positions"] = [VocabFrequency.from_dict(
            d) for d in src.get("positions", [])]
        return VocabWord(**src)


@dataclass
class Vocabulary:
    word_list: List[str] = field(default_factory=list)
    word_index: Dict[str, int] = field(default_factory=dict)
    word_count: Dict[str, int] = field(default_factory=dict)
    words: List[VocabWord] = field(default_factory=list)

    def add_word(self, word: Token) -> None:
        pass

    def to_dict(self):
        dst = dict(vars(self).items())
        dst["words"] = [sec.to_dict() for sec in self.words]
        return dst

    @staticmethod
    def from_dict(src: Dict[str, Any]) -> "Vocabulary":
        for key in ["words"]:
            src[key] = [VocabWord.from_dict(d) for d in src.get(key, [])]
        return Vocabulary(**src)


@dataclass
class BodySentence:
    index: int = field(default=0)
    value: str = field(default="")
    rubifuri: str = field(default="")
    words_with_pos: List[VocabPartOfSpeech] = field(default_factory=list)
    sub_phrases: List[str] = field(default_factory=list)

    def get_sentence(self) -> str:
        return self.value

    def get_words(self) -> List[str]:
        return [part_.word for part_ in self.words_with_pos]

    def get_pos(self) -> List[VocabPartOfSpeech]:
        return [part for part in self.words_with_pos]

    def find_pos_by_word(self, word: str) -> List[VocabPartOfSpeech]:
        return [part for part in self.words_with_pos if part.word == word]

    def to_dict(self) -> Dict[str, int | str | List[Dict[str, str]]]:
        dst = dict(vars(self).items())
        dst["words_with_pos"] = [pos.to_dict() for pos in self.words_with_pos]
        return dst

    @staticmethod
    def from_dict(src: Dict[str, Any]) -> "BodySentence":
        for key in ["words_with_pos"]:
            src[key] = [VocabPartOfSpeech.from_dict(
                d) for d in src.get(key, [])]
        return BodySentence(**src)

    @property
    def nbr_of_tokens(self) -> int:
        return len(self.words_with_pos)


@dataclass
class BodyParagraph:
    index: int = field(default=0)
    sentences: List[BodySentence] = field(default_factory=list)

    def get_sentence(self, sentence_index: int) -> str:
        return self.sentences[sentence_index].get_sentence()

    def get_sentences(self) -> List[str]:
        return [sent.get_sentence() for sent in self.sentences]

    def get_words(self) -> List[List[str]]:
        return [sent.get_words() for sent in self.sentences]

    def get_pos(self) -> List[List[VocabPartOfSpeech]]:
        return [sent.get_pos() for sent in self.sentences]

    def find_pos_by_word(self, word: str) -> List[List[VocabPartOfSpeech]]:
        return [sent.find_pos_by_word(word) for sent in self.sentences]

    def to_dict(self) -> Dict[str, int | List[Dict[str, int | str | List[Dict[str, str]]]]]:
        dst = dict(vars(self).items())
        dst["sentences"] = [sec.to_dict() for sec in self.sentences]
        return dst

    @staticmethod
    def from_dict(src: Dict[str, Any]) -> "BodyParagraph":
        for key in ["sentences"]:
            src[key] = [BodySentence.from_dict(
                d) for d in src.get(key, [])]
        return BodyParagraph(**src)

    @property
    def nbr_of_sentences(self) -> int:
        return len(self.sentences)

    @property
    def nbr_of_tokens_per_sentence(self) -> List[int]:
        return [sent.nbr_of_tokens for sent in self.sentences]

    @property
    def nbr_of_tokens(self) -> int:
        return sum(self.nbr_of_tokens_per_sentence)


@dataclass
class BodySection:
    index: int = field(default=0)
    paragraph: List[BodyParagraph] = field(default_factory=list)

    def get_sentence(self, paragraph_index: int, sentence_index: int) -> str:
        return self.paragraph[paragraph_index].get_sentence(sentence_index)

    def get_sentences(self) -> List[List[str]]:
        return [para.get_sentences() for para in self.paragraph]

    def get_words(self) -> List[List[str]]:
        return [para.get_words() for para in self.paragraph]

    def get_pos(self) -> List[List[List[VocabPartOfSpeech]]]:
        return [para.get_pos() for para in self.paragraph]

    def find_pos_by_word(self, word: str) -> List[List[List[VocabPartOfSpeech]]]:
        return [para.find_pos_by_word(word) for para in self.paragraph]

    def to_dict(self) -> Dict[str, int | List[Dict[str, int | List[Dict[str, int | str | List[Dict[str, str]]]]]]]:
        dst = dict(vars(self).items())
        dst["paragraph"] = [sec.to_dict() for sec in self.paragraph]
        return dst

    @staticmethod
    def from_dict(src: Dict[str, Any]) -> "BodySection":
        for key in ["paragraph"]:
            src[key] = [BodyParagraph.from_dict(
                d) for d in src.get(key, [])]
        return BodySection(**src)

    @property
    def nbr_of_paragraph(self) -> int:
        return len(self.paragraph)

    @property
    def nbr_of_tokens_per_paragraph(self) -> List[int]:
        return [para.nbr_of_tokens for para in self.paragraph]

    @property
    def nbr_of_tokens(self) -> int:
        return sum(self.nbr_of_tokens_per_paragraph)


@dataclass
class BodyChapter:
    index: int = field(default=0)
    sections: List[BodySection] = field(default_factory=list)

    def get_sentence(self, section_index: int, paragraph_index: int, sentence_index: int) -> str:
        return self.sections[section_index].get_sentence(paragraph_index, sentence_index)

    def get_sentences(self) -> List[List[List[str]]]:
        return [sec.get_sentences() for sec in self.sections]

    def get_words(self):
        return [sec.get_words() for sec in self.sections]

    def get_pos(self) -> List[List[List[List[VocabPartOfSpeech]]]]:
        return [sec.get_pos() for sec in self.sections]

    def find_pos_by_word(self, word: str) -> List[List[List[List[VocabPartOfSpeech]]]]:
        return [sec.find_pos_by_word(word) for sec in self.sections]

    def to_dict(self) -> Dict[str, int | List[Dict[str, int | List[Dict[str, int | List[Dict[str, int | str | List[Dict[str, str]]]]]]]]]:
        dst = dict(vars(self).items())
        dst["sections"] = [sec.to_dict() for sec in self.sections]
        return dst

    @staticmethod
    def from_dict(src: Dict[str, Any]) -> "BodyChapter":
        for key in ["sections"]:
            src[key] = [BodySection.from_dict(
                d) for d in src.get(key, [])]
        return BodyChapter(**src)

    @property
    def nbr_of_tokens_per_section(self) -> List[int]:
        return [sec.nbr_of_tokens for sec in self.sections]

    @property
    def nbr_of_tokens(self) -> int:
        return sum(self.nbr_of_tokens_per_section)


@dataclass
class Body:
    chapters: List[BodyChapter] = field(default_factory=list)

    def get_sentence(
        self, chapter_index: int, section_index: int,
        paragraph_index: int, sentence_index: int
    ) -> str:
        return self.chapters[chapter_index].get_sentence(
            section_index, paragraph_index, sentence_index
        )

    def get_sentences(self) -> List[List[List[str]]]:
        return [chap.get_sentences() for chap in self.chapters]

    def get_words(self):
        return [chap.get_words() for chap in self.chapters]

    def get_pos(self) -> List[List[List[List[VocabPartOfSpeech]]]]:
        return [chap.get_pos() for chap in self.chapters]

    def find_pos_by_word(self, word: str) -> List[List[List[List[VocabPartOfSpeech]]]]:
        return [chap.find_pos_by_word(word) for chap in self.chapters]

    def to_dict(self):
        dst = dict(vars(self).items())
        dst["chapters"] = [chap.to_dict() for chap in self.chapters]
        return dst

    @staticmethod
    def from_dict(src: Dict[str, Any]) -> "Body":
        for key in ["chapters"]:
            src[key] = [BodyChapter.from_dict(
                d) for d in src.get(key, [])]
        return Body(**src)

    @property
    def nbr_of_tokens_per_chapter(self) -> List[int]:
        return [chap.nbr_of_tokens for chap in self.chapters]

    @property
    def nbr_of_tokens(self) -> int:
        return sum(self.nbr_of_tokens_per_chapter)


@dataclass
class Document:
    fpath: str = field(default="")
    document_id: int = field(default=0)
    bib_info: BibInfo = field(default_factory=BibInfo)
    toc: Toc = field(default_factory=Toc)
    body: Body = field(default_factory=Body)
    vocab: Vocabulary = field(default_factory=Vocabulary)

    def get_sentence(
        self,
        chapter_index: int,
        section_index: int,
        paragraph_index: int,
        sentence_index: int
    ) -> str:
        return self.body.get_sentence(
            chapter_index,
            section_index,
            paragraph_index,
            sentence_index
        )

    def to_dict(self):
        dst = dict(vars(self).items())
        dst["bib_info"] = self.bib_info.to_dict()
        dst["toc"] = self.toc.to_dict()
        dst["body"] = self.body.to_dict()
        dst["vocab"] = self.vocab.to_dict()
        return dst

    @staticmethod
    def from_dict(src: Dict[str, Any]) -> "Document":
        src["bib_info"] = BibInfo.from_dict(src.get("bib_info", {}))
        src["toc"] = Toc.from_dict(src.get("toc", {}))
        src["body"] = Body.from_dict(src.get("body", {}))
        src["vocab"] = Vocabulary.from_dict(src.get("vocab", {}))
        return Document(**src)


def get_sub_phrase(sentence: str):
    dst = []
    doc = nlp(sentence)
    for sent in doc.sents:
        for token in ginza.bunsetu_head_tokens(sent):
            for _, sub_phrase in ginza.sub_phrases(token):
                dst.append(str(sub_phrase).replace("+", ""))
    return dst


def extract_pos(text: str) -> List[VocabPartOfSpeech]:
    doc_: Doc = nlp(text.strip())
    return [VocabPartOfSpeech.from_token(token) for token in doc_]


def generate_body(fpath: str, threshold_nbr_of_tokens: int) -> Body:
    chapters = []
    with open(fpath, "r", encoding="utf-8") as ff:
        sections = []
        paragraph_list = []
        sentences = []
        paragraph_index = 0
        nbr_of_tokens = 0
        for line_id, line in enumerate(ff):
            parts_of_speech = extract_pos(line)
            sub_phrase_list = get_sub_phrase(line)
            sentence = BodySentence(
                line_id, line.strip(),
                words_with_pos=parts_of_speech,
                sub_phrases=sub_phrase_list
            )
            sentences.append(sentence)
            nbr_of_tokens += sentence.nbr_of_tokens
            if nbr_of_tokens > threshold_nbr_of_tokens:
                paragraph = BodyParagraph(paragraph_index, sentences)
                paragraph_list.append(paragraph)
                sentences = []
                nbr_of_tokens = 0
        if nbr_of_tokens > 0:
            paragraph = BodyParagraph(paragraph_index, sentences)
            paragraph_list.append(paragraph)
        section = BodySection(0, paragraph_list)
        sections.append(section)
        chapter = BodyChapter(0, sections)
        chapters.append(chapter)

    body = Body(chapters)
    return body
