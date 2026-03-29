#!/usr/bin/perl
#lab2-word-counter.pl
use warnings;
use strict;

my $tot=0;

while (<>) {
  while (/[a-zA-Z]+/g) {
    $tot++;
  }
}

print "\nTotal number of words: $tot\n";