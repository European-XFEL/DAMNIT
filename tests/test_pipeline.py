from textwrap import dedent

import pytest

from damnit.context import Variable, Group, Pipeline
from damnit.ctxsupport import damnit_ctx as ctxmod

from .helpers import mkcontext


def test_pipeline_execute_with_data(mock_run):
    @Variable
    def a(run):
        return 1

    @Variable(data="proc")
    def b(run):
        return 2

    pipe = Pipeline(run_data="raw").add(a, b)
    res = pipe.with_context(data=mock_run, proposal=123, run_number=1).execute()
    assert res.cells["a"].data == 1
    assert "b" not in res.cells


def test_pipeline_default_merges_autodiscovery():
    code = """
    from damnit_ctx import Variable, Group, Pipeline

    @Variable
    def base(run):
        return 1

    @Group
    class G:
        @Variable
        def val(self, run):
            return 2

    g = G()
    Pipeline.default().add(G(name="extra"))
    """
    ctx = mkcontext(code)
    assert "base" in ctx.vars
    assert "g.val" in ctx.vars
    assert "extra.val" in ctx.vars


def test_pipeline_set_default_override():
    code = """
    from damnit_ctx import Variable, Pipeline

    @Variable
    def base(run):
        return 1

    @Variable
    def chosen(run):
        return 2

    pipe = Pipeline()
    pipe.add(chosen)
    Pipeline.set_default(pipe)
    """
    ctx = mkcontext(code)
    assert set(ctx.vars) == {"chosen"}


def test_pipeline_from_context_file(tmp_path, mock_run):
    code = dedent("""
        from damnit_ctx import Variable

        @Variable
        def foo(run):
            return 7
    """)
    ctx_path = tmp_path / "context.py"
    ctx_path.write_text(code)

    pipe = Pipeline.from_context_file(ctx_path)
    res = pipe.with_context(data=mock_run, proposal=1, run_number=1).execute()
    assert res.cells["foo"].data == 7


def test_pipeline_nested_context_isolation():
    outer_code = dedent("""
        from damnit_ctx import Variable, Pipeline
        from damnit.context import Pipeline as PipelinePublic

        @Variable
        def outer(run):
            return 1

        inner_code = '''
        from damnit_ctx import Variable, Pipeline

        @Variable
        def inner(run):
            return 2

        pipe = Pipeline()
        pipe.add(inner)
        Pipeline.set_default(pipe)
        '''

        PipelinePublic.from_str(inner_code)
        Pipeline.default().add(outer)
    """)
    pipe = Pipeline.from_str(outer_code)
    assert "outer" in pipe.vars
    assert "inner" not in pipe.vars


def test_pipeline_state_reset_after_exception():
    failing_code = dedent("""
        from damnit_ctx import Variable, Pipeline

        @Variable
        def bad(run):
            return 1

        Pipeline.default().add(bad)
        raise RuntimeError("boom")
    """)
    with pytest.raises(RuntimeError):
        Pipeline.from_str(failing_code)

    pipe = Pipeline.default()
    ctx = pipe.compile()
    assert "bad" not in ctx.vars
    ctxmod._DEFAULT_PIPELINE_STATE.set(None)


def test_pipeline_sequential_contexts_no_leak():
    code_a = dedent("""
        from damnit_ctx import Variable, Pipeline

        @Variable
        def a(run):
            return 1

        pipe = Pipeline()
        pipe.add(a)
        Pipeline.set_default(pipe)
    """)
    code_b = dedent("""
        from damnit_ctx import Variable, Pipeline

        @Variable
        def b(run):
            return 2

        pipe = Pipeline()
        pipe.add(b)
        Pipeline.set_default(pipe)
    """)
    pipe_a = Pipeline.from_str(code_a)
    pipe_b = Pipeline.from_str(code_b)
    assert set(pipe_a.vars) == {"a"}
    assert set(pipe_b.vars) == {"b"}
