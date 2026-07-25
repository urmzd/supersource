// 05-strings-and-runes: a Go string is a read-only byte slice that everyone
// treats as a sequence of characters. Indexing gives bytes, ranging gives
// runes, and len gives neither of the things you usually want.
package main

import (
	"fmt"
	"sort"
	"strings"
	"unicode/utf8"
)

func main() {
	s := "héllo"
	fmt.Println("1.", len(s), utf8.RuneCountInString(s))
	fmt.Println("2.", s[1], s[0])

	fmt.Print("3.")
	for i, r := range s {
		fmt.Printf(" %d:%c", i, r)
	}
	fmt.Println()

	fmt.Println("4.", len([]rune(s)), len([]byte(s)))
	// Slicing by byte offset can cut a rune in half.
	fmt.Printf("5. %q %q\n", s[0:2], s[0:3])

	// strings.Title-style byte slicing is where mojibake comes from.
	fmt.Println("6.", strings.ToUpper(s), len(strings.ToUpper(s)))

	// A map has no order, so anything printed from one must be sorted first.
	counts := map[string]int{"b": 2, "a": 1, "c": 3}
	keys := make([]string, 0, len(counts))
	for k := range counts {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	fmt.Println("7.", keys)

	missing, ok := counts["zzz"]
	fmt.Println("8.", missing, ok, len(counts))

	// Deleting during iteration is legal; reading a deleted key gives zero.
	delete(counts, "a")
	fmt.Println("9.", counts["a"], len(counts))

	// Strings are immutable, so every "modification" allocates.
	t := s
	t += "!"
	fmt.Println("10.", s, t)
}
