"""unittest"""

import unittest
from document_py import document


class DocumentTest(unittest.TestCase):
    def test_author_from_dict(self):
        src = {"name": "test"}
        expected = document.Author(**src)
        self.assertEqual(document.Author.from_dict(src), expected)

    def test_bib_info_from_dict_wo_authors(self):
        src = {
            "id": 0, "title": "test", "title_en": "test_en",
            "publisher": "test_pbu", "publication_date": "2024-10-12"
        }
        expected = document.BibInfo(**src)
        self.assertEqual(document.BibInfo.from_dict(src), expected)

    def test_bib_info_from_dict_w_authors(self):
        src = {
            "id": 0, "title": "test", "title_en": "test_en",
            "publisher": "test_pbu", "publication_date": "2024-10-12"
        }
        authors = [
            document.Author(name="test1"),
            document.Author(name="test2")
        ]
        translators = [
            document.Author(name="hoge"),
            document.Author(name="fuga")
        ]
        expected = document.BibInfo(**src)
        expected.authors = authors
        expected.translators = translators
        src["authors"] = [author.to_dict() for author in authors]
        src["translators"] = [author.to_dict() for author in translators]
        self.assertEqual(document.BibInfo.from_dict(src), expected)

    def test_toc_section_from_dict(self):
        src = {"index": 0, "title": "test"}
        expected = document.TocSection(**src)
        self.assertEqual(document.TocSection.from_dict(src), expected)

    def test_toc_chapter_from_dict_wo_sections(self):
        src = {"index": 0, "title": "test"}
        expected = document.TocChapter(**src)
        self.assertEqual(document.TocChapter.from_dict(src), expected)

    def test_toc_chapter_from_dict_w_sections(self):
        src = {"index": 0, "title": "test"}
        sections = [
            document.TocSection(index=0, title="test3"),
            document.TocSection(index=1, title="test4")
        ]
        expected = document.TocChapter(**src)
        expected.sections = sections
        src["sections"] = [
            section.to_dict() for section in sections
        ]
        self.assertEqual(document.TocChapter.from_dict(src), expected)

    def test_toc_from_dict(self):
        src = {"index": 0, "title": "test"}
        sections = [
            document.TocSection(index=0, title="test3"),
            document.TocSection(index=1, title="test4")
        ]
        chapter = document.TocChapter(**src)
        chapter.sections = sections
        expected = document.Toc(chapters=[chapter])
        src = {"chapters": [chapter.to_dict()]}
        self.assertEqual(document.Toc.from_dict(src), expected)

    def test_vocab_pos_from_dict(self):
        src = {
            "word": "test", "lemma": "NOUN",
            "pos": "名詞", "detail_1": "普通名詞",
            "detail_2": "test2"
        }
        expected = document.VocabPartOfSpeech(**src)
        self.assertEqual(
            document.VocabPartOfSpeech.from_dict(src),
            expected
        )


if __name__ == "__main__":
    unittest.main()
