"""Tests for BDD engine correctness."""


class TestBddMatchesExplicit:
    def test_counter(self, counter_explicit, counter_bdd):
        assert counter_bdd.total_reachable == len(counter_explicit.reachable_states)

    def test_gcd(self, gcd_explicit, gcd_bdd):
        assert gcd_bdd.total_reachable == len(gcd_explicit.reachable_states)

    def test_mult(self, mult_explicit, mult_bdd):
        assert mult_bdd.total_reachable == len(mult_explicit.reachable_states)

    def test_mutex(self, mutex_explicit, mutex_bdd):
        assert mutex_bdd.total_reachable == len(mutex_explicit.reachable_states)


class TestBddEncoding:
    def test_counter_bits(self, counter_bdd):
        enc = counter_bdd.encoding
        assert enc["mode"].n_bits == 1
        assert enc["press"].n_bits == 1
        assert enc["x"].n_bits == 5  # ceil(log2(26))

    def test_mutex_bits(self, mutex_bdd):
        enc = mutex_bdd.encoding
        assert enc["process1"].n_bits == 2  # 3 values -> 2 bits
        assert enc["turn"].n_bits == 1      # 2 values -> 1 bit
        assert enc["flag1"].n_bits == 1     # boolean -> 1 bit


class TestFixpointConvergence:
    def test_counter_converges(self, counter_bdd):
        assert counter_bdd.iterations[-1].new_states_count == 0

    def test_gcd_converges(self, gcd_bdd):
        assert gcd_bdd.iterations[-1].new_states_count == 0

    def test_mult_converges(self, mult_bdd):
        assert mult_bdd.iterations[-1].new_states_count == 0

    def test_mutex_converges(self, mutex_bdd):
        assert mutex_bdd.iterations[-1].new_states_count == 0
