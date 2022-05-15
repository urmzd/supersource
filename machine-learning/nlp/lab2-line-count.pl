#!/usr/bin/perl

my $fname = shift;
chomp $fname;

my $cnt = `wc -l < $fname`;
chomp $cnt;

print "$fname has $cnt lines\n";
