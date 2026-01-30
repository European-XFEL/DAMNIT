import pytest

from damnit.context import GroupError
from .helpers import mkcontext


def test_group_expands_variables_and_prefixes():
    code = """
    from damnit_ctx import Variable, Group

    @Group(tags=["XGM"])
    class XGM:
        device_name: str

        @Variable(title="Pulse Energy", tags=["Energy"])
        def pulse_energy(self, run):
            return 1

    xgm = XGM(name="xgm_sa2", title="XGM Diag", device_name="SA2_XTD6_XGM/XGM/DOOCS")
    """
    ctx = mkcontext(code)
    assert "xgm_sa2.pulse_energy" in ctx.vars
    var = ctx.vars["xgm_sa2.pulse_energy"]
    assert var.title == "XGM Diag/Pulse Energy"
    assert var.tags == {"XGM", "Energy"}


def test_group_linking_resolves_dependencies():
    code = """
    from damnit_ctx import Variable, Group

    @Group
    class Inner:
        @Variable
        def energy(self, run):
            return 1

    @Group
    class Outer:
        inner: Inner

        @Variable
        def scaled(self, run, energy: "self#inner.energy"):
            return energy * 2

    outer = Outer(name="outer", inner=Inner(name="inner"))
    """
    ctx = mkcontext(code)
    assert "inner.energy" in ctx.vars
    assert "outer.scaled" in ctx.vars
    assert ctx.vars["outer.scaled"].arg_dependencies() == {"energy": "inner.energy"}


def test_group_variable_field_linking(mock_run):
    code = """
    from damnit_ctx import Variable, Group

    @Variable
    def base(run):
        return 1

    @Group
    class A:
        v: Variable

        @Variable
        def asdf(self, run, data: "self#v"):
            return data + 1

    a1 = A(name="a1", v=base)
    a2 = A(name="a2", v=a1.asdf)
    """
    ctx = mkcontext(code)
    results = ctx.execute(mock_run, 1000, 123, {})
    assert results.cells["base"].data == 1
    assert results.cells["a1.asdf"].data == 2
    assert results.cells["a2.asdf"].data == 3
    assert ctx.vars["a1.asdf"].arg_dependencies() == {"data": "base"}
    assert ctx.vars["a2.asdf"].arg_dependencies() == {"data": "a1.asdf"}


def test_group_nested_variable_field_chain(mock_run):
    code = """
    from damnit_ctx import Variable, Group

    @Variable
    def base(run):
        return 5

    @Group
    class Leaf:
        linked_base: Variable
        scale: int = 2

        @Variable
        def scaled(self, run, value: "self#linked_base"):
            return value * self.scale

    @Group
    class Mid:
        leaf: Leaf
        offset: int = 4

        @Variable
        def total(self, run, value: "self#leaf.scaled"):
            return value + self.offset

    @Group
    class Outer:
        mid: Mid
        factor: int = 2

        @Variable
        def final(self, run, value: "self#mid.total"):
            return value * self.factor

        @Variable
        def extra(self, run, value: "self#mid.leaf.linked_base"):
            return 2 * value

    leaf = Leaf(name="leaf", linked_base=base, scale=2)
    mid = Mid(name="mid", leaf=leaf, offset=4)
    outer = Outer(name="outer", mid=mid, factor=2)
    """
    ctx = mkcontext(code)
    results = ctx.execute(mock_run, 1000, 123, {})
    assert results.cells["base"].data == 5
    assert results.cells["leaf.scaled"].data == 10
    assert results.cells["mid.total"].data == 14
    assert results.cells["outer.final"].data == 28
    assert results.cells["outer.extra"].data == 10
    assert ctx.vars["leaf.scaled"].arg_dependencies() == {"value": "base"}
    assert ctx.vars["mid.total"].arg_dependencies() == {"value": "leaf.scaled"}
    assert ctx.vars["outer.final"].arg_dependencies() == {"value": "mid.total"}
    assert ctx.vars["outer.extra"].arg_dependencies() == {"value": "base"}


def test_group_name_from_context_assignment(mock_run):
    code = """
    from damnit_ctx import Variable, Group

    @Group
    class G:
        @Variable
        def val(self, run):
            return 3

    group_name = G()
    """
    ctx = mkcontext(code)
    results = ctx.execute(mock_run, 1000, 123, {})
    assert "group_name.val" in ctx.vars
    assert ctx.vars["group_name.val"].title == "group_name/val"
    assert results.cells["group_name.val"].data == 3


def test_group_name_instance(mock_run):
    code = """
    from damnit_ctx import Variable, Group

    @Group
    class G:
        @Variable
        def val(self, run):
            return 5

    g = G(name="instance")
    """
    ctx = mkcontext(code)
    results = ctx.execute(mock_run, 1000, 123, {})
    assert "instance.val" in ctx.vars
    assert ctx.vars["instance.val"].title == "instance/val"
    assert "decorated.val" not in ctx.vars
    assert results.cells["instance.val"].data == 5


def test_group_optional_component_drops_missing_dependency(mock_run):
    code = """
    from damnit_ctx import Variable, Group

    @Group
    class A:
        @Variable
        def var(self, run):
            return 41

    @Group
    class B:
        upstream: A | None = None

        @Variable
        def needs_upstream(self, run, value: "self#upstream.var"):
            return value + 1

        @Variable
        def optional_upstream(self, run, value: "self#upstream.var" = 42):
            return value + 1

    b = B(name="b")
    """
    ctx = mkcontext(code)
    results = ctx.execute(mock_run, 1000, 123, {})
    assert "b.needs_upstream" not in ctx.vars
    assert "b.optional_upstream" in ctx.vars
    assert ctx.vars["b.optional_upstream"].arg_dependencies() == {}
    assert results.cells["b.optional_upstream"].data == 43


def test_group_execute_intra_dependencies_and_overrides(mock_run):
    code = """
    from damnit_ctx import Variable, Group

    @Group(tags=["Default"])
    class G:
        offset: float = 0.0

        @Variable(title="Base", tags=["Base"])
        def base(self, run):
            return 10

        @Variable(title="Adjusted")
        def adjusted(self, run, base: "self#base"):
            return base + self.offset

    g = G(name="g", title="MyDiag", offset=5, tags=["Override"])
    """
    ctx = mkcontext(code)
    results = ctx.execute(mock_run, 1000, 123, {})

    assert results.cells["g.base"].data == 10
    assert results.cells["g.adjusted"].data == 15
    assert ctx.vars["g.adjusted"].arg_dependencies() == {"base": "g.base"}
    assert ctx.vars["g.adjusted"].title == "MyDiag/Adjusted"
    assert ctx.vars["g.base"].tags == {"Override", "Base"}
    assert ctx.vars["g.adjusted"].tags == {"Override"}


def test_group_linking_exec_and_dependency_paths(mock_run):
    code = """
    from damnit_ctx import Variable, Group

    @Group
    class XGM:
        offset: float = 0.0

        @Variable(title="Energy")
        def energy(self, run):
            return 10

        @Variable(title="Corrected")
        def corrected(self, run, energy: "self#energy"):
            return energy + self.offset

    @Group
    class MID:
        xgm: XGM
        factor: float = 2.0

        @Variable(title="Scaled")
        def scaled(self, run, energy: "self#xgm.corrected"):
            return energy * self.factor

    xgm_sa2 = XGM(name="xgm_sa2", title="XGM", offset=1.5)
    mid = MID(name="mid", title="MID", sep=" | ", xgm=xgm_sa2, factor=3)
    """
    ctx = mkcontext(code)
    results = ctx.execute(mock_run, 1000, 123, {})

    assert results.cells["xgm_sa2.energy"].data == 10
    assert results.cells["xgm_sa2.corrected"].data == pytest.approx(11.5)
    assert results.cells["mid.scaled"].data == pytest.approx(34.5)
    assert ctx.vars["mid.scaled"].arg_dependencies() == {"energy": "xgm_sa2.corrected"}
    assert ctx.vars["mid.scaled"].title == "MID | Scaled"


def test_group_optional_component_cascades_drop():
    code = """
    from typing import Optional
    from damnit_ctx import Variable, Group

    @Group
    class Upstream:
        @Variable
        def value(self, run):
            return 1

    @Group
    class Outer:
        upstream: Optional[Upstream] = None

        @Variable
        def needs_upstream(self, run, value: "self#upstream.value"):
            return value + 1

        @Variable
        def depends_on_needs(self, run, value: "self#needs_upstream"):
            return value + 1

    outer = Outer(name="outer")
    """
    ctx = mkcontext(code)
    assert "outer.needs_upstream" not in ctx.vars
    assert "outer.depends_on_needs" not in ctx.vars


def test_group_reserved_field_annotation_rejected():
    code = """
    from damnit_ctx import Variable, Group

    @Group
    class Bad:
        name: str

        @Variable
        def val(self, run):
            return 1

    bad = Bad(name="bad")
    """
    with pytest.raises(GroupError):
        mkcontext(code)


def test_group_reserved_attribute_rejected():
    code = """
    from damnit_ctx import Variable, Group

    @Group
    class Bad:
        tags = ["nope"]

        @Variable
        def val(self, run):
            return 1

    bad = Bad(name="bad")
    """
    with pytest.raises(GroupError):
        mkcontext(code)


def test_group_missing_name_for_nested_component_raises():
    code = """
    from damnit_ctx import Variable, Group

    @Group
    class Inner:
        @Variable
        def val(self, run):
            return 1

    @Group
    class Outer:
        inner: Inner

        @Variable
        def out(self, run, val: "self#inner.val"):
            return val + 1

    outer = Outer(name="outer", inner=Inner())
    """
    with pytest.raises(GroupError):
        mkcontext(code)


def test_group_inheritance_and_reset(mock_run):
    code = """
    from damnit_ctx import Variable, Group

    @Group(tags=["BaseTag"])
    class Base:
        @Variable(title="BaseVar")
        def base(self, run):
            return 5

    @Group
    class Derived(Base):
        @Variable(title="ChildVar")
        def child(self, run, base: "self#base"):
            return base + 1

    @Group(tags=["AltTag"])
    class DerivedAlt(Base):
        @Variable(title="AltVar")
        def alt(self, run):
            return 7

    derived = Derived(name="derived")
    alt = DerivedAlt(name="alt", title="Alt")
    """
    ctx = mkcontext(code)
    results = ctx.execute(mock_run, 1000, 123, {})

    assert results.cells["derived.base"].data == 5
    assert results.cells["derived.child"].data == 6
    assert results.cells["alt.base"].data == 5
    assert results.cells["alt.alt"].data == 7

    assert ctx.vars["derived.base"].title == "derived/BaseVar"
    assert ctx.vars["derived.child"].title == "derived/ChildVar"
    assert ctx.vars["derived.base"].tags == {"BaseTag"}

    assert ctx.vars["alt.base"].title == "Alt/BaseVar"
    assert ctx.vars["alt.alt"].title == "Alt/AltVar"
    assert ctx.vars["alt.base"].tags == {"AltTag"}


def test_group_inherited_dataclass_fields(mock_run):
    code = """
    from damnit_ctx import Variable, Group

    @Group
    class A:
        label: str

        @Variable
        def label_len(self, run):
            return len(self.label)

    @Group
    class B(A):
        age: int

        @Variable
        def age_plus(self, run):
            return self.age + 1

    b = B(label="asdf", age=12)
    """
    ctx = mkcontext(code)
    results = ctx.execute(mock_run, 1000, 123, {})
    assert results.cells["b.label_len"].data == 4
    assert results.cells["b.age_plus"].data == 13


def test_group_decorated_subclass_non_default_field(mock_run):
    code = """
    from damnit_ctx import Variable, Group

    @Group
    class A:
        base: int = 1

        @Variable
        def base_val(self, run):
            return self.base

    @Group
    class B(A):
        value: int

        @Variable
        def combined(self, run):
            return self.base + self.value

    b = B(value=3)
    """
    with pytest.raises(TypeError):
        mkcontext(code)
    # results = ctx.execute(mock_run, 1000, 123, {})
    # assert results.cells["b.base_val"].data == 1
    # assert results.cells["b.combined"].data == 4


def test_group_duplicate_names_raise():
    code = """
    from damnit_ctx import Variable, Group

    @Group
    class G:
        @Variable
        def val(self, run):
            return 1

    g1 = G(name="dup")
    g2 = G(name="dup")
    """
    with pytest.raises(GroupError):
        mkcontext(code)


def test_group_ambiguous_assignment_raises():
    code = """
    from damnit_ctx import Variable, Group

    @Group
    class G:
        @Variable
        def val(self, run):
            return 1

    g = G()
    alias = g
    """
    with pytest.raises(GroupError):
        mkcontext(code)
