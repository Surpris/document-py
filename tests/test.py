"""unittest"""

import unittest
from document_py import document


class DocumentTest(unittest.TestCase):
    def test_author_from_dict(self):
        src = {"name": "test"}
        expected = document.Author(**src)
        self.assertEqual(document.Author.from_dict(src), expected)

    def test_bib_info_from_dict_wo_authors(self):
        src = {"id": 0, "title": "test", "title_en": "test_en",
               "publisher": "test_pbu", "publication_date": "2024-10-12"}
        expected = document.BibInfo(**src)
        self.assertEqual(document.BibInfo.from_dict(src), expected)

    def test_bib_info_from_dict_w_authors(self):
        src = {"id": 0, "title": "test", "title_en": "test_en",
               "publisher": "test_pbu", "publication_date": "2024-10-12"}
        authors = [document.Author(name="test1"), document.Author(name="test2")]
        translators = [document.Author(name="hoge"), document.Author(name="fuga")]
        expected = document.BibInfo(**src)
        expected.authors = authors
        expected.translators = translators
        src["authors"] = [author.to_dict() for author in authors]
        src["translators"] = [author.to_dict() for author in translators]
        self.assertEqual(document.BibInfo.from_dict(src), expected)


if __name__ == "__main__":
    unittest.main()
