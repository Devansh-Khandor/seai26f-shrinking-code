import unittest

from harness.reply import ReplyParseError, parse_reply

PRIMARY = ("Here you go.\n\nFILE: skill_01.py\n```python\nprint('hi')\n```\n\n"
           "FILE: lib/util.py\n```python\nX = 1\n```\n")


class TestReply(unittest.TestCase):
    def test_primary(self):
        r = parse_reply(PRIMARY)
        self.assertEqual(r.files, {"skill_01.py": "print('hi')\n", "lib/util.py": "X = 1\n"})
        self.assertEqual(r.deletes, ())

    def test_fallback_info_string(self):
        self.assertEqual(parse_reply("```lua skill_02.lua\nprint(1)\n```").files,
                         {"skill_02.lua": "print(1)\n"})

    def test_delete(self):
        r = parse_reply("DELETE: old.py\nFILE: skill_02.py\n```py\npass\n```")
        self.assertEqual(r.deletes, ("old.py",))
        self.assertEqual(r.files, {"skill_02.py": "pass\n"})

    def test_prose_only_fails(self):
        with self.assertRaises(ReplyParseError):
            parse_reply("I think you should use a dictionary.")

    def test_path_escape_rejected(self):
        for bad in ("../x.py", "/etc/x.py", "C:\\x.py", "a/../../x.py"):
            with self.assertRaises(ReplyParseError, msg=bad):
                parse_reply(f"FILE: {bad}\n```python\npass\n```")

    def test_unclosed_fence_fails(self):
        with self.assertRaises(ReplyParseError):
            parse_reply("FILE: skill_01.py\n```python\nprint(1)\n")

    def test_bold_or_backticked_file_label(self):
        r = parse_reply("**FILE: `skill_01.py`**\n```python\nx = 1\n```")
        self.assertEqual(r.files, {"skill_01.py": "x = 1\n"})

    def test_plain_fence_without_path_ignored(self):
        r = parse_reply("Example:\n```\nnot a file\n```\nFILE: a.py\n```python\nx = 1\n```")
        self.assertEqual(r.files, {"a.py": "x = 1\n"})


class TestReplyVariants(unittest.TestCase):
    def files(self, text):
        return list(parse_reply(text).files)

    def test_leading_underscores_kept(self):
        self.assertEqual(self.files("FILE: __init__.py\n```python\nx = 1\n```"), ["__init__.py"])
        self.assertEqual(self.files("FILE: `_util.lua`\n```lua\nx = 1\n```"), ["_util.lua"])

    def test_trailing_annotation_dropped(self):
        self.assertEqual(self.files("FILE: skill_01.py (new)\n```python\nx = 1\n```"), ["skill_01.py"])
        self.assertEqual(self.files("FILE: `skill_01.py` (updated)\n```python\nx = 1\n```"), ["skill_01.py"])

    def test_bold_label_only(self):
        self.assertEqual(self.files("**FILE:** `skill_01.py`\n```python\nx = 1\n```"), ["skill_01.py"])

    def test_fence_info_variants(self):
        for info in ("skill_01.py", "python:skill_01.py", 'python title="skill_01.py"'):
            self.assertEqual(self.files(f"```{info}\nx = 1\n```"), ["skill_01.py"], info)

    def test_heading_label(self):
        self.assertEqual(self.files("### skill_01.py\n```python\nx = 1\n```"), ["skill_01.py"])

    def test_bad_paths_are_parse_errors(self):
        for bad in ("./", ".", "a\x00b.py", "x" * 300 + ".py", "a:b.py", "con.py", "a<b.py"):
            with self.assertRaises(ReplyParseError, msg=repr(bad)):
                parse_reply(f"FILE: {bad}\n```python\npass\n```")

    def test_think_block_ignored(self):
        text = "<think>\nFILE: draft.py\n```python\nx = 0\n```\n</think>\nFILE: skill_01.py\n```python\nx = 1\n```"
        self.assertEqual(self.files(text), ["skill_01.py"])

    def test_none_is_parse_error(self):
        with self.assertRaises(ReplyParseError):
            parse_reply(None)


if __name__ == "__main__":
    unittest.main()
