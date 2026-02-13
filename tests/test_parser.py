"""Tests for SMV parser correctness."""
from smvis.smv_model import BoolType, EnumType, RangeType, IntLit


class TestCounterParser:
    def test_variable_count(self, counter_model):
        assert len(counter_model.variables) == 3

    def test_variable_names(self, counter_model):
        assert set(counter_model.variables.keys()) == {"mode", "press", "x"}

    def test_mode_is_enum(self, counter_model):
        vd = counter_model.variables["mode"]
        assert isinstance(vd.var_type, EnumType)
        assert set(vd.var_type.values) == {"off", "on"}

    def test_press_is_bool(self, counter_model):
        assert isinstance(counter_model.variables["press"].var_type, BoolType)

    def test_x_is_range(self, counter_model):
        vd = counter_model.variables["x"]
        assert isinstance(vd.var_type, RangeType)
        assert vd.var_type.lo == 0
        assert vd.var_type.hi == 25

    def test_defines(self, counter_model):
        assert "count_max" in counter_model.defines
        assert isinstance(counter_model.defines["count_max"], IntLit)
        assert counter_model.defines["count_max"].value == 10

    def test_init_count(self, counter_model):
        assert len(counter_model.inits) == 2
        assert "mode" in counter_model.inits
        assert "x" in counter_model.inits

    def test_next_count(self, counter_model):
        assert len(counter_model.nexts) == 2
        assert "mode" in counter_model.nexts
        assert "x" in counter_model.nexts

    def test_fairness(self, counter_model):
        assert len(counter_model.fairness) == 1

    def test_spec_counts(self, counter_model):
        by_kind = {}
        for sp in counter_model.specs:
            by_kind[sp.kind] = by_kind.get(sp.kind, 0) + 1
        assert by_kind.get("INVARSPEC", 0) == 5
        assert by_kind.get("LTLSPEC", 0) == 17
        assert by_kind.get("CTLSPEC", 0) == 4


class TestGcdParser:
    def test_variable_count(self, gcd_model):
        assert len(gcd_model.variables) == 3

    def test_variable_names(self, gcd_model):
        assert set(gcd_model.variables.keys()) == {"a", "b", "pc"}

    def test_a_range(self, gcd_model):
        vd = gcd_model.variables["a"]
        assert isinstance(vd.var_type, RangeType)
        assert vd.var_type.lo == 0
        assert vd.var_type.hi == 10

    def test_pc_is_enum(self, gcd_model):
        vd = gcd_model.variables["pc"]
        assert isinstance(vd.var_type, EnumType)
        assert len(vd.var_type.values) == 5

    def test_init_only_pc(self, gcd_model):
        assert len(gcd_model.inits) == 1
        assert "pc" in gcd_model.inits

    def test_spec_counts(self, gcd_model):
        by_kind = {}
        for sp in gcd_model.specs:
            by_kind[sp.kind] = by_kind.get(sp.kind, 0) + 1
        assert by_kind.get("CTLSPEC", 0) == 4
        assert by_kind.get("LTLSPEC", 0) == 7
        assert by_kind.get("INVARSPEC", 0) == 3


class TestMultParser:
    def test_variable_count(self, mult_model):
        assert len(mult_model.variables) == 4

    def test_variable_names(self, mult_model):
        assert set(mult_model.variables.keys()) == {"a", "b", "prod", "pc"}

    def test_prod_range(self, mult_model):
        vd = mult_model.variables["prod"]
        assert isinstance(vd.var_type, RangeType)
        assert vd.var_type.lo == 0
        assert vd.var_type.hi == 100


class TestMutexParser:
    def test_variable_count(self, mutex_model):
        assert len(mutex_model.variables) == 5

    def test_variable_names(self, mutex_model):
        assert set(mutex_model.variables.keys()) == {
            "process1", "process2", "turn", "flag1", "flag2"
        }

    def test_all_inits(self, mutex_model):
        assert len(mutex_model.inits) == 5

    def test_all_nexts(self, mutex_model):
        assert len(mutex_model.nexts) == 5

    def test_nextref_in_flag1(self, mutex_model):
        from smvis.explicit_engine import _find_next_refs
        refs = _find_next_refs(mutex_model.nexts["flag1"])
        assert "process1" in refs

    def test_nextref_in_flag2(self, mutex_model):
        from smvis.explicit_engine import _find_next_refs
        refs = _find_next_refs(mutex_model.nexts["flag2"])
        assert "process2" in refs
