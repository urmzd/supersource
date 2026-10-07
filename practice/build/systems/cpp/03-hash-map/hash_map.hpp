// hash_map.hpp - an open-addressed hash map with a std::unordered_map-shaped
// interface.
//
// You write the marked regions. main.cpp tests them.
//
// std::unordered_map is specified as separate chaining, which is why its
// iterators stay valid across a rehash and why it is slower than it looks:
// every lookup chases a pointer to a separately allocated node. This one uses
// open addressing, which is faster and invalidates everything on rehash. The
// interface therefore documents weaker guarantees than the standard's, and
// noticing that difference is most of the exercise.
#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

template <typename K, typename V, typename Hash = std::hash<K>>
class HashMap {
 public:
  using key_type = K;
  using mapped_type = V;
  using value_type = std::pair<const K, V>;
  using size_type = std::size_t;

  HashMap() = default;
  explicit HashMap(size_type initial_buckets) { rehash(initial_buckets); }

  // ---- capacity ---------------------------------------------------------

  size_type size() const noexcept { return size_; }
  bool empty() const noexcept { return size_ == 0; }
  size_type bucket_count() const noexcept { return slots_.size(); }

  float load_factor() const noexcept {
    return slots_.empty() ? 0.0f
                          : static_cast<float>(size_) / static_cast<float>(slots_.size());
  }

  // ---- lookup -----------------------------------------------------------

  V* find(const K& key) {
    // SOLUTION-BEGIN
    if (slots_.empty()) return nullptr;
    size_type i;
    if (!probe(key, i)) return nullptr;
    return &slots_[i].kv->second;
    // SOLUTION-END
  }

  const V* find(const K& key) const {
    return const_cast<HashMap*>(this)->find(key);
  }

  bool contains(const K& key) const { return find(key) != nullptr; }

  V& at(const K& key) {
    V* p = find(key);
    if (!p) throw std::out_of_range("HashMap::at");
    return *p;
  }

  // Default-constructs a value when the key is absent, like the standard's.
  V& operator[](const K& key) {
    // SOLUTION-BEGIN
    return insert_or_find(key, V{}).first->second;
    // SOLUTION-END
  }

  // ---- modifiers --------------------------------------------------------

  // Returns {pointer to the element, whether it was newly inserted}.
  std::pair<value_type*, bool> insert(const K& key, const V& value) {
    // SOLUTION-BEGIN
    return insert_or_find(key, value);
    // SOLUTION-END
  }

  // Overwrites an existing value rather than leaving it alone.
  std::pair<value_type*, bool> insert_or_assign(const K& key, const V& value) {
    // SOLUTION-BEGIN
    auto [slot, inserted] = insert_or_find(key, value);
    if (!inserted) slot->second = value;
    return {slot, inserted};
    // SOLUTION-END
  }

  bool erase(const K& key) {
    // SOLUTION-BEGIN
    if (slots_.empty()) return false;
    size_type i;
    if (!probe(key, i)) return false;

    // Destroy the element but leave a TOMBSTONE, not an empty slot. Clearing
    // to empty would truncate every probe run passing through here, making
    // later entries unreachable while they still occupy space.
    slots_[i].kv.reset();
    slots_[i].state = State::Tombstone;
    --size_;
    return true;
    // SOLUTION-END
  }

  void clear() noexcept {
    for (Slot& s : slots_) {
      s.kv.reset();
      s.state = State::Empty;
    }
    size_ = 0;
    used_ = 0;
  }

  void rehash(size_type n) {
    // SOLUTION-BEGIN
    // Round up to a power of two so the bucket index is a mask rather than a
    // modulo. Never shrink below what is needed to hold the live elements at
    // a sane load factor.
    size_type want = n < size_ * 2 ? size_ * 2 : n;
    size_type cap = 8;
    while (cap < want) cap *= 2;
    if (cap == slots_.size() && used_ == size_) return;

    std::vector<Slot> fresh(cap);
    for (Slot& s : slots_) {
      if (s.state != State::Full) continue;
      size_type i = Hash{}(s.kv->first) & (cap - 1);
      while (fresh[i].state == State::Full) i = (i + 1) & (cap - 1);
      fresh[i].kv = std::move(s.kv);
      fresh[i].state = State::Full;
    }
    slots_ = std::move(fresh);
    used_ = size_;  // rehashing is what reclaims tombstones
    // SOLUTION-END
  }

  // ---- iteration --------------------------------------------------------
  //
  // A forward iterator that skips empty and tombstoned slots. Order is
  // unspecified and deliberately not stable across modification.
  class iterator {
   public:
    iterator(HashMap* m, size_type i) : m_(m), i_(i) { skip_to_live(); }

    value_type& operator*() const { return *m_->slots_[i_].kv; }
    value_type* operator->() const { return m_->slots_[i_].kv.get(); }

    iterator& operator++() {
      ++i_;
      skip_to_live();
      return *this;
    }

    bool operator==(const iterator& o) const { return i_ == o.i_; }
    bool operator!=(const iterator& o) const { return i_ != o.i_; }

   private:
    void skip_to_live() {
      while (i_ < m_->slots_.size() && m_->slots_[i_].state != State::Full) ++i_;
    }
    HashMap* m_;
    size_type i_;
  };

  iterator begin() { return iterator(this, 0); }
  iterator end() { return iterator(this, slots_.size()); }

 private:
  enum class State : unsigned char { Empty, Full, Tombstone };

  // The pair is heap-allocated because value_type's key is const, which makes
  // the pair non-assignable and therefore awkward to hold directly in a vector
  // that needs to move elements around during rehash. A production
  // implementation would use aligned storage and placement new; this keeps the
  // exercise about probing rather than about manual object lifetime, which is
  // exercise 01's job.
  struct Slot {
    std::unique_ptr<value_type> kv;
    State state = State::Empty;
  };

  // Find `key`, or the slot it should occupy. True when found.
  bool probe(const K& key, size_type& idx) const {
    // SOLUTION-BEGIN
    const size_type mask = slots_.size() - 1;
    size_type i = Hash{}(key) & mask;
    size_type first_tomb = slots_.size();

    for (size_type step = 0; step <= mask; ++step) {
      const Slot& s = slots_[i];
      if (s.state == State::Empty) {
        // A probe run ends at the first genuinely empty slot. Prefer an
        // earlier tombstone for insertion so that repeated insert/erase
        // cycles do not lengthen probe runs without bound.
        idx = (first_tomb == slots_.size()) ? i : first_tomb;
        return false;
      }
      if (s.state == State::Tombstone) {
        if (first_tomb == slots_.size()) first_tomb = i;
      } else if (s.kv->first == key) {
        idx = i;
        return true;
      }
      i = (i + 1) & mask;
    }
    idx = (first_tomb == slots_.size()) ? 0 : first_tomb;
    return false;
    // SOLUTION-END
  }

  std::pair<value_type*, bool> insert_or_find(const K& key, const V& value) {
    // SOLUTION-BEGIN
    if (slots_.empty()) rehash(8);

    size_type i;
    if (probe(key, i)) return {slots_[i].kv.get(), false};

    // Rehash at 3/4 of slots USED, counting tombstones. Using the live count
    // alone lets a delete-heavy workload fill the table with tombstones while
    // reporting a low load factor, and probe runs decay towards linear.
    //
    // The target size comes from the LIVE count, not from the current table
    // size, so a table that tripped this check purely on tombstones is rebuilt
    // at the same capacity with the tombstones dropped. Asking for double the
    // current size instead makes an insert/erase loop grow the table forever
    // while it holds nothing.
    if ((used_ + 1) * 4 >= slots_.size() * 3) {
      rehash((size_ + 1) * 2);
      probe(key, i);  // the old index refers to the old table
    }

    if (slots_[i].state == State::Empty) ++used_;  // a tombstone was already counted
    slots_[i].kv = std::make_unique<value_type>(key, value);
    slots_[i].state = State::Full;
    ++size_;
    return {slots_[i].kv.get(), true};
    // SOLUTION-END
  }

  std::vector<Slot> slots_;
  size_type size_ = 0;  // live elements
  size_type used_ = 0;  // live elements plus tombstones
};
