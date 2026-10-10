#!/usr/bin/perl
# @author: Urmzd Mukhammadnaim
# @file: a1q5.pl
# @date: 28/09/21
# @description: Solution for assignment 1 question 5
# @course: CSCI 4152

use strict;
use warnings;

local $/;
my $stdin = <>;

# Remove \n on last 'line'.
chomp $stdin;
# Get all tags.
my @allTags = $stdin =~ /(<(?:[^>]|\n)*>)/gm;
# Filter out comments.
my @nonComments = grep { $_ !~ /(<!--)/gm } @allTags;

# Map to count.
my @tagLengths = map { length $_ } @nonComments;
my $numberOfTags = scalar @tagLengths;
my @sortedLengths = sort {$a <=> $b} @tagLengths;
my $minLength = $sortedLengths[0] || 0;
my $maxLength = $sortedLengths[-1] || 0;

# length : -2 
# total : -1
sub average {
  my @values = @_;
  my $total = $values[-2]; 
  my $length = $values[-1];

  my $str = join ",", @values;

  if (scalar @values == 2 ) {
    return $total / $length; 
  }

  return average(@values[1..$#values - 2], $total + $values[0] , $length); 
}

my $averageLength = average(@sortedLengths, 0, scalar @sortedLengths);

print "Tag count: $numberOfTags\n";
print "Min length: $minLength\n";
print "Max length: $maxLength\n";
printf "Avg length: %.2f\n", $averageLength; 
