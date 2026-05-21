"""Unit tests for SWEBenchVerifiedTask._normalize_django_directive.

The function converts SWE-bench's unittest verbose format
("test_method (module.ClassName)") to Django runtests.py format
("module.ClassName.test_method").

The bug fixed here: when class_path already ends with ".method_name"
(i.e. the entry was stored as "test_foo (module.Class.test_foo)"),
the old code produced "module.Class.test_foo.test_foo" — a double-appended
method name that Django rejects, causing reward=0 for otherwise-correct patches.
"""

from __future__ import annotations


from swebench_verified_cube.task import SWEBenchVerifiedTask

_normalize = SWEBenchVerifiedTask._normalize_django_directive


class TestNormalizeDjangoDirective:
    def test_standard_format(self) -> None:
        """Normal case: method_name (module.ClassName) → module.ClassName.method_name."""
        assert _normalize("test_foo (myapp.tests.MyTest)") == "myapp.tests.MyTest.test_foo"

    def test_already_normalized(self) -> None:
        """If class_path already ends with the method name, don't duplicate it."""
        assert _normalize("test_foo (myapp.tests.MyTest.test_foo)") == "myapp.tests.MyTest.test_foo"

    def test_already_normalized_nested(self) -> None:
        """Deeper nesting: method appears both in path and as final segment."""
        assert _normalize("test_bar (a.b.c.MyClass.test_bar)") == "a.b.c.MyClass.test_bar"

    def test_different_suffix_not_stripped(self) -> None:
        """class_path ends with a different method — should still append."""
        assert _normalize("test_foo (myapp.tests.MyTest.test_bar)") == "myapp.tests.MyTest.test_bar.test_foo"

    def test_passthrough_when_no_parens(self) -> None:
        """Dotted path with no parens is passed through unchanged."""
        assert _normalize("myapp.tests.MyTest.test_foo") == "myapp.tests.MyTest.test_foo"

    def test_malformed_returns_none(self) -> None:
        """Strings with whitespace/special chars but no valid format return None,
        so _build_test_cmd's `valid = [t for t in normalized if t is not None]`
        filter can drop them."""
        assert _normalize("some weird string") is None

    def test_strips_surrounding_whitespace(self) -> None:
        """Leading/trailing whitespace in directive is stripped before matching."""
        assert _normalize("  test_foo (myapp.tests.MyTest)  ") == "myapp.tests.MyTest.test_foo"

    def test_single_class_no_module(self) -> None:
        """Simple class with no module prefix."""
        assert _normalize("test_foo (MyTest)") == "MyTest.test_foo"
