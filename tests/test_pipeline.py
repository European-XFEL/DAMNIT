from textwrap import dedent

import h5py
import pytest

from damnit.context import Variable, Group, Pipeline
from damnit_ctx import _DEFAULT_PIPELINE_STATE,  GroupError, RunData

from .helpers import mkcontext


def test_pipeline_execute_with_data(mock_run):
    @Variable
    def a(run):
        return 1

    @Variable(data="proc")
    def b(run):
        return 2

    pipe = Pipeline().add(a, b)
    res = pipe.with_context(data=mock_run, proposal=123, run_number=1).select(run_data="raw").execute()
    assert res.cells["a"].data == 1
    assert "b" not in res.cells


def test_pipeline_execute_data_override(mock_run):
    override = {}

    @Variable
    def used_override(run, _override=override):
        return run is _override

    pipe = Pipeline().add(used_override)
    res = pipe.with_context(data=mock_run, proposal=1, run_number=1).execute(data=override)
    assert bool(res.cells["used_override"].data) is True


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
    assert "bad" not in pipe.vars
    _DEFAULT_PIPELINE_STATE.set(None)


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


def test_pipeline_add_accepts_nested_sequences_and_rejects_invalid_iterables():
    @Variable
    def a(run):
        return 1

    @Variable
    def b(run):
        return 2

    pipe = Pipeline().add([a, (b,)])
    assert set(pipe.vars) == {"a", "b"}

    with pytest.raises(TypeError, match=r"Pipeline\.add accepts"):
        Pipeline().add(123)

    def gen():
        yield a

    with pytest.raises(TypeError, match=r"Pipeline\.add accepts"):
        Pipeline().add(gen())


def test_pipeline_add_rejects_string():
    with pytest.raises(TypeError, match=r"Pipeline\.add accepts"):
        Pipeline().add("not-a-variable")


def test_pipeline_add_invalidates_compiled_context():
    @Variable
    def a(run):
        return 1

    @Variable
    def b(run):
        return 2

    pipe = Pipeline().add(a)
    assert set(pipe.vars) == {"a"}
    pipe.add(b)
    assert set(pipe.vars) == {"a", "b"}


def test_pipeline_set_default_rejects_non_pipeline():
    with pytest.raises(TypeError, match="expects a Pipeline instance"):
        Pipeline.set_default("not a pipeline")


def test_pipeline_build_context_rejects_duplicate_variable_names():
    def f(run):
        return 1

    v1 = Variable(f)
    v2 = Variable(f)

    with pytest.raises(GroupError, match=r"Duplicate variable name 'f'"):
        Pipeline().add(v1, v2)


def test_pipeline_build_context_rejects_unnamed_group_instance():
    @Group
    class G:
        @Variable
        def val(self, run):
            return 1

    with pytest.raises(GroupError, match=r"has no name"):
        Pipeline().add(G())


def test_pipeline_build_context_rejects_duplicate_group_names():
    @Group
    class G:
        @Variable
        def val(self, run):
            return 1

    g1 = G(name="same")
    g2 = G(name="same")

    with pytest.raises(GroupError, match=r"Group name 'same' is used by multiple group instances"):
        Pipeline().add(g1, g2)


def test_pipeline_with_context_updates():
    input_vars = {"x": 1}
    data_obj = object()
    pipe = Pipeline(name="a", proposal=1, run_number=2, data=data_obj, input_vars=input_vars)
    new_pipe = pipe.with_context(name="b", proposal=3, input_vars={"y": 2})

    assert (pipe.name, pipe.proposal, pipe.run_number, pipe.data, pipe.input_vars) == (
        "a", 1, 2, data_obj, {"x": 1}
    )
    assert (new_pipe.name, new_pipe.proposal, new_pipe.run_number, new_pipe.data, new_pipe.input_vars) == (
        "b", 3, 2, data_obj, {"y": 2}
    )

    input_vars["x"] = 999
    assert pipe.input_vars == {"x": 1}


def test_pipeline_select_filters():
    code = dedent("""
        from damnit_ctx import Variable

        @Variable(title="Alpha", data="raw")
        def a(run):
            return 1

        @Variable(title="Beta", data="proc", cluster=True)
        def b(run):
            return 2

        @Variable(title="Gamma", data="raw")
        def c(run, x: "var#a"):
            return x + 1
    """)
    pipe = Pipeline.from_str(code)

    only_c = pipe.select(variables=("c",))
    assert set(only_c.vars) == {"a", "c"}  # dependency is kept

    by_match = pipe.select(match=("beta",))
    assert set(by_match.vars) == {"b"}

    clustered = pipe.select(cluster=True)
    assert set(clustered.vars) == {"b"}

    raw_only = pipe.select(run_data="raw")
    assert set(raw_only.vars) == {"a", "c"}


def test_pipeline_select_then_add_does_not_resurrect_filtered_vars():
    @Variable
    def a(run):
        return 1

    @Variable
    def b(run):
        return 2

    @Variable
    def c(run):
        return 3

    selected = Pipeline().add(a, b).select(variables=("a",))
    assert set(selected.vars) == {"a"}

    selected.add(c)
    assert set(selected.vars) == {"a", "c"}


def test_pipeline_select_run_data_overrides_execution(mock_run):
    @Variable(data="raw")
    def raw_var(run):
        return 1

    @Variable(data="proc")
    def proc_var(run):
        return 2

    pipe = Pipeline().add(raw_var, proc_var)
    selected = pipe.select(run_data="raw")
    res = selected.with_context(data=mock_run, proposal=1, run_number=1).execute()
    assert "raw_var" in res.cells
    assert "proc_var" not in res.cells


def test_pipeline_execute_requires_data_or_proposal_run_number():
    @Variable
    def a(run):
        return 1

    with pytest.raises(ValueError, match="proposal and run_number must be set"):
        Pipeline().add(a).execute()


def test_pipeline_execute(mock_run):
    code = dedent("""
        from damnit_ctx import Variable

        @Variable
        def val(run, x: "input#x"):
            return x
    """)
    pipe = Pipeline.from_str(code).with_context(
        data=mock_run,
        proposal=1,
        run_number=1,
        input_vars={"x": 1},
    )

    assert pipe.results is None
    res = pipe.execute(input_vars={"x": 2})
    assert res.cells["val"].data == 2


def test_pipeline_vars_to_dict():
    code = dedent("""
        from damnit_ctx import Variable

        @Variable
        def keep(run):
            return 1

        @Variable(transient=True)
        def transient_var(run):
            return 2
    """)
    pipe = Pipeline.from_str(code)
    assert set(pipe.vars_to_dict()) == {"keep"}
    assert set(pipe.vars_to_dict(inc_transient=True)) == {"keep", "transient_var"}


def test_pipeline_save_hdf5(tmp_path, mock_run):
    code = dedent("""
        from damnit_ctx import Variable

        @Variable
        def a(run):
            return 1
    """)
    pipe = Pipeline.from_str(code).with_context(data=mock_run, proposal=1, run_number=1)

    with pytest.raises(RuntimeError, match="Call execute\\(\\) first"):
        pipe.save_hdf5(tmp_path / "out.h5")

    pipe.execute()
    
    result_h5_file = tmp_path / 'out.h5'
    pipe.save_hdf5(result_h5_file)

    with h5py.File(result_h5_file) as f:
        assert f['a/data'][()] == 1

    pipe.save_hdf5(result_h5_file, reduced_only=True)

    with h5py.File(result_h5_file) as f:
        assert 'a/data' not in f
        assert f['.reduced/a'][()] == 1


def test_pipeline_copy():
    @Variable
    def a(run):
        return 1

    @Variable
    def b(run):
        return 2

    pipe = Pipeline(input_vars={"x": 1}).add(a)

    clone = pipe.copy()
    clone.input_vars["x"] = 2
    clone.input_vars["y"] = 3
    clone.add(b)

    assert pipe.input_vars == {"x": 1}
    assert clone.input_vars == {"x": 2, "y": 3}
    assert set(pipe.vars) == {"a"}
    assert set(clone.vars) == {"a", "b"}


def test_reuse_vars(mock_run):
    code = """
    from damnit_ctx import Variable, Group, Pipeline

    @Group
    class A:
        value: int = 1

        @Variable
        def var(self, run):
            return self.value * run['data']

    @Variable
    def source(run):
        pipe = Pipeline(proposal=1, run_number=2, data={'data': 10})
        pipe.add(A(value=2, name='aa'))
        pipe.execute()
        return pipe.results.cells['aa.var'].data

    @Variable
    def target(run, data: 'var#source', run_number: 'meta#run_number'):
        return data + run_number
    """
    ctx = mkcontext(code)
    res = ctx.execute(mock_run, 1, 1, {})
    
    assert set(ctx.vars) == {'source', 'target'}
    assert res.cells['source'].data == 20
    assert res.cells['target'].data == 21
