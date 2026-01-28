#!/usr/bin/perl
# Letter frequencies

#fill in the blanks _______________
#the required output: letters of the input text 
#with their frequencies and normalized frequencies 
#ordered from the most frequent to the least frequent

use warnings;
use strict;

my %f=();
my $tot=0;

while (<>) {
    foreach my $let (/\w+/g) {              
         $f{lc $let} += 1;                                     
         $tot ++;                                  
     }
}


print "10 most common words are:\n";
my @sorted_words = sort { $f{$b} cmp $f{$a} } keys %f;

my $no_of_single_words = 0;
for my $i (0..$#sorted_words) {
    my $word = $sorted_words[$i];
    if ($i < 9) {
        print "$word "; 
    }

    if ($i == 10) {
        print "$word\n";
    }

    if ($f{$word} == 1) {
        $no_of_single_words += 1;
    }
}

print "The number of hapax legomena is $no_of_single_words \n";
