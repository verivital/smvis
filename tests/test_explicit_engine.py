"""Tests for explicit-state engine correctness."""


class TestCounterExplicit:
    def test_total_states(self, counter_explicit):
        assert counter_explicit.total_states == 104  # 2*2*26

    def test_initial_count(self, counter_explicit):
        assert len(counter_explicit.initial_states) == 2

    def test_reachable_count(self, counter_explicit):
        assert len(counter_explicit.reachable_states) == 24

    def test_transition_count(self, counter_explicit):
        assert len(set(counter_explicit.transitions)) == 48

    def test_initial_are_reachable(self, counter_explicit):
        for s in counter_explicit.initial_states:
            assert s in counter_explicit.reachable_states

    def test_state_to_dict_complete(self, counter_explicit):
        for s in counter_explicit.reachable_states:
            assert s in counter_explicit.state_to_dict


class TestGcdExplicit:
    def test_total_states(self, gcd_explicit):
        assert gcd_explicit.total_states == 605  # 11*11*5

    def test_initial_count(self, gcd_explicit):
        assert len(gcd_explicit.initial_states) == 121  # 11*11, pc=l1

    def test_reachable_count(self, gcd_explicit):
        assert len(gcd_explicit.reachable_states) == 352

    def test_transition_count(self, gcd_explicit):
        assert len(set(gcd_explicit.transitions)) == 352


class TestMultExplicit:
    def test_total_states(self, mult_explicit):
        # 11*11*101*4 = 48884
        assert mult_explicit.total_states == 48884

    def test_initial_count(self, mult_explicit):
        assert len(mult_explicit.initial_states) == 121  # 11*11

    def test_reachable_count(self, mult_explicit):
        assert len(mult_explicit.reachable_states) == 1902

    def test_transition_count(self, mult_explicit):
        assert len(set(mult_explicit.transitions)) == 1902


class TestMutexExplicit:
    def test_total_states(self, mutex_explicit):
        assert mutex_explicit.total_states == 72  # 3*3*2*2*2

    def test_initial_count(self, mutex_explicit):
        assert len(mutex_explicit.initial_states) == 1

    def test_reachable_count(self, mutex_explicit):
        assert len(mutex_explicit.reachable_states) == 16

    def test_transition_count(self, mutex_explicit):
        assert len(set(mutex_explicit.transitions)) == 30


class TestTransitionConsistency:
    def test_counter_transitions_in_reachable(self, counter_explicit):
        for src, dst in counter_explicit.transitions:
            assert src in counter_explicit.reachable_states
            assert dst in counter_explicit.reachable_states

    def test_mutex_transitions_in_reachable(self, mutex_explicit):
        for src, dst in mutex_explicit.transitions:
            assert src in mutex_explicit.reachable_states
            assert dst in mutex_explicit.reachable_states
