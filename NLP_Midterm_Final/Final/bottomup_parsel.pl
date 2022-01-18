:-op(1000, xfy, ==>).
 
%% Lexicon
n ==> [they].
n ==> [homework].
n ==> [park].
d ==> [the].
d ==> [the].
v ==> [did].
p ==> [in].
 
%% Phrase Structure Rules
s ==> dp, vp.
vp ==> v, dp.
vp ==> vp, pp.
pp ==> p,dp.
dp ==> d, np.
dp ==> np.
np ==> n.

%% Shift-Reduce Parser
% Base
sr_parse([s], []).

% Shift
sr_parse(Stack, [Word|Words]):-
   (Cat ==> [Word]),
   sr_parse([Cat|Stack], Words).

% Reduce
sr_parse([Y,X|Rest], String):-
   (Z ==> X, Y),
   sr_parse([Z|Rest], String).

sr_parse([X|Rest], String):-
   (Y ==> X),
   sr_parse([Y|Rest], String).
