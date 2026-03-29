#!/usr/bin/perl

use warnings;
use strict;

sub conc {
  my ($a, $b) = @_;
  my @con = ($a, $b);
  my ($a1, $b1) = sort @con;
  return $a1.$b1;
}

print &conc("aaa","ccc");
print "\n";
print &conc("ccc","aaa");
print "\n";
