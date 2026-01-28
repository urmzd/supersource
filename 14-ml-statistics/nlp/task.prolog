isChild(john, mary).
isChild(ann, john).
isChild(tom, john).
isGrandChild(X, Y) :-
    isChild(X, Z),
    isChild(Z, Y).