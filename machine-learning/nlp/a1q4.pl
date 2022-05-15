#!/usr/bin/perl
# @author: Urmzd Mukhammadnaim
# @file: a1q4.pl
# @date: 28/09/21
# @description: Solution for Assignment 1 Question 4
# @course: CSCI 4152

use strict;
use warnings;

# Read from standard in.
while (my $line = <>) {
  # Strip line.
  chomp $line;
  # Extract email. 
  my ($email) = 
  ($line =~/([a-zA-Z]+(\w|-|=|\.|\+|_)*@\w+\2*\.(\w|-|\.|\+|_)*[a-zA-z])/);

  # Default value to empty string.
  $email //= '';

  # Print nothing if no match, print email and line otherwise.
  if ($email eq '') {
    print "$email \n"; 
  } else {
    print "$email: $line \n"
  }
}
