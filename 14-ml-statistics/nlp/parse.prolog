s(S, R) :-
    np(S, I),
    vp(I, R).

np(S, R) :-
    d(S, I),
    np(I, R).

d([the|R], R).
n([dog|R], R).
n([dogs|R], R).
vp([run|R], R).
vp([runs|R], R).

