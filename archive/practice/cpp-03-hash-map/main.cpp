// main.cpp - the tests. Given to you; do not edit them to make them pass.
#include "hash_map.hpp"

#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

static int checks = 0;
#define CHECK(cond, what)                                                     \
  do {                                                                        \
    ++checks;                                                                 \
    if (!(cond)) {                                                            \
      std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, (what));   \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

// A hash that sends everything to the same bucket. Every collision-handling
// path is exercised on every operation, which is the only reliable way to test
// probing: with a good hash, collisions are rare enough to hide bugs.
struct AlwaysCollide {
  std::size_t operator()(int) const noexcept { return 0; }
};

// A hash with only two distinct outputs, for a slightly less degenerate case.
struct TwoBuckets {
  std::size_t operator()(int k) const noexcept { return static_cast<std::size_t>(k % 2); }
};

static void test_basics() {
  HashMap<std::string, int> m;
  CHECK(m.empty(), "a fresh map is empty");
  CHECK(m.size() == 0, "size 0");
  CHECK(m.find("absent") == nullptr, "find on an empty map returns null");
  CHECK(!m.contains("absent"), "contains agrees");

  auto [p1, inserted1] = m.insert("alpha", 1);
  CHECK(inserted1, "a new key reports inserted");
  CHECK(p1->second == 1, "with the right value");
  CHECK(m.size() == 1, "size grew");

  auto [p2, inserted2] = m.insert("alpha", 99);
  CHECK(!inserted2, "an existing key reports not-inserted");
  CHECK(p2->second == 1, "and insert does NOT overwrite");
  CHECK(m.size() == 1, "size unchanged");

  auto [p3, inserted3] = m.insert_or_assign("alpha", 99);
  CHECK(!inserted3, "insert_or_assign reports not-inserted for an existing key");
  CHECK(p3->second == 99, "but does overwrite");
  CHECK(*m.find("alpha") == 99, "and the change is visible");

  CHECK(m.at("alpha") == 99, "at returns the value");
  bool threw = false;
  try {
    (void)m.at("missing");
  } catch (const std::out_of_range&) {
    threw = true;
  }
  CHECK(threw, "at throws for a missing key");
}

static void test_subscript_default_constructs() {
  HashMap<std::string, int> m;
  CHECK(m["fresh"] == 0, "operator[] default-constructs a missing value");
  CHECK(m.size() == 1, "and inserts it");

  m["fresh"] = 5;
  CHECK(*m.find("fresh") == 5, "assigning through operator[] works");
  CHECK(m.size() == 1, "without inserting again");

  m["other"] += 3;
  CHECK(*m.find("other") == 3, "compound assignment on a fresh key starts from zero");
}

static void test_many_keys_survive_rehashing() {
  HashMap<int, int> m;
  const int N = 5000;
  for (int i = 0; i < N; ++i) {
    auto [p, inserted] = m.insert(i, i * 3);
    CHECK(inserted, "each distinct key is a new insertion");
    CHECK(p->second == i * 3, "with its value");
  }
  CHECK(m.size() == N, "every key counted once");
  CHECK(m.bucket_count() >= N, "the table grew");
  CHECK(m.load_factor() <= 0.8f, "and kept the load factor sane");

  for (int i = 0; i < N; ++i) {
    const int* v = m.find(i);
    CHECK(v != nullptr, "every key is still findable after many rehashes");
    CHECK(*v == i * 3, "with the right value");
  }
  CHECK(m.find(N) == nullptr, "a key never inserted is absent");
}

// The test that separates a working open-addressed map from a broken one.
// With every key in one bucket, erasing from the middle of a probe run must
// leave a tombstone or the keys behind it become unreachable.
static void test_erase_from_the_middle_of_a_probe_run() {
  HashMap<int, int, AlwaysCollide> m;
  for (int i = 0; i < 10; ++i) m.insert(i, i * 100);
  CHECK(m.size() == 10, "ten colliding keys all stored");

  CHECK(m.erase(5), "erasing an existing key reports true");
  CHECK(m.size() == 9, "size dropped by one");
  CHECK(m.find(5) == nullptr, "the erased key is gone");

  // Everything else must still be reachable, especially the keys that were
  // placed AFTER the erased one in the probe run.
  for (int i = 0; i < 10; ++i) {
    if (i == 5) continue;
    const int* v = m.find(i);
    CHECK(v != nullptr, "keys after the hole are still reachable");
    CHECK(*v == i * 100, "with their values intact");
  }

  CHECK(!m.erase(5), "erasing twice reports false");
  CHECK(m.insert(5, 555).second, "the slot can be reused");
  CHECK(*m.find(5) == 555, "and the reinserted key is found");
  CHECK(m.size() == 10, "back to ten");
}

// Tombstones cost probe steps but hold nothing, so a table that resizes on the
// live count alone degrades to a linear scan under churn while reporting a low
// load factor.
static void test_churn_does_not_degrade() {
  HashMap<int, int> m;
  for (int round = 0; round < 20000; ++round) {
    m.insert(round, round);
    CHECK(m.erase(round), "erase succeeds during churn");
  }
  CHECK(m.size() == 0, "churn leaves nothing behind");
  CHECK(m.bucket_count() < 4096, "and the table did not grow without bound");

  // The map must still work afterwards.
  m.insert(1, 1);
  CHECK(*m.find(1) == 1, "the map still works after heavy churn");
}

static void test_collisions_with_a_terrible_hash() {
  HashMap<int, int, AlwaysCollide> m;
  const int N = 500;
  for (int i = 0; i < N; ++i) m.insert(i, i);
  CHECK(m.size() == N, "every colliding key stored");
  for (int i = 0; i < N; ++i) {
    const int* v = m.find(i);
    CHECK(v != nullptr && *v == i, "every colliding key is retrievable");
  }

  // Erase every other key, then check the survivors.
  for (int i = 0; i < N; i += 2) CHECK(m.erase(i), "erase succeeds");
  CHECK(m.size() == N / 2, "half remain");
  for (int i = 1; i < N; i += 2) {
    CHECK(m.find(i) != nullptr, "odd keys survived the erasures");
  }
  for (int i = 0; i < N; i += 2) {
    CHECK(m.find(i) == nullptr, "even keys are gone");
  }
}

static void test_two_bucket_hash() {
  HashMap<int, int, TwoBuckets> m;
  for (int i = 0; i < 200; ++i) m.insert(i, i * 7);
  for (int i = 0; i < 200; ++i) {
    const int* v = m.find(i);
    CHECK(v != nullptr && *v == i * 7, "a two-bucket hash still stores everything");
  }
}

static void test_iteration_visits_everything_once() {
  HashMap<int, int> m;
  const int N = 300;
  for (int i = 0; i < N; ++i) m.insert(i, i * 2);
  m.erase(7);
  m.erase(100);
  m.erase(299);

  std::set<int> seen;
  int count = 0;
  for (auto& kv : m) {
    CHECK(kv.second == kv.first * 2, "iteration yields matching key/value pairs");
    CHECK(seen.insert(kv.first).second, "no key is visited twice");
    ++count;
  }
  CHECK(count == N - 3, "iteration visits exactly the live elements");
  CHECK(seen.count(7) == 0, "and skips erased ones");
  CHECK(seen.count(100) == 0, "all of them");
  CHECK(seen.count(299) == 0, "including the last");

  HashMap<int, int> empty;
  int n = 0;
  for (auto& kv : empty) {
    (void)kv;
    ++n;
  }
  CHECK(n == 0, "iterating an empty map yields nothing");
}

static void test_clear_and_reuse() {
  HashMap<std::string, int> m;
  for (int i = 0; i < 100; ++i) m.insert("key" + std::to_string(i), i);
  CHECK(m.size() == 100, "populated");

  m.clear();
  CHECK(m.size() == 0, "clear empties the map");
  CHECK(m.empty(), "and it reports empty");
  CHECK(m.find("key50") == nullptr, "the entries are gone");

  m.insert("fresh", 1);
  CHECK(m.size() == 1, "the map is reusable after clear");
  CHECK(*m.find("fresh") == 1, "and works");
}

// Values must be the objects that were stored, not copies that drifted.
static void test_values_are_mutable_in_place() {
  HashMap<int, std::string> m;
  m.insert(1, "one");
  std::string* v = m.find(1);
  CHECK(v != nullptr, "found");
  *v = "ONE";
  CHECK(*m.find(1) == "ONE", "mutating through the returned pointer changes the map");

  m.at(1) += "!";
  CHECK(*m.find(1) == "ONE!", "at returns a mutable reference");
}

int main() {
  test_basics();
  test_subscript_default_constructs();
  test_many_keys_survive_rehashing();
  test_erase_from_the_middle_of_a_probe_run();
  test_churn_does_not_degrade();
  test_collisions_with_a_terrible_hash();
  test_two_bucket_hash();
  test_iteration_visits_everything_once();
  test_clear_and_reuse();
  test_values_are_mutable_in_place();
  std::printf("ok  cpp/03-hash-map  %d checks passed\n", checks);
  return 0;
}
