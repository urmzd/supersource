module tinyllm

go 1.22

// The vendored contracts module (DESIGN 2.15). Unused until a unit imports
// it; the replace is ignored until then, so this file never needs a rewrite.
replace supersource.urmzd.com/tl/contracts => ../contracts/go
