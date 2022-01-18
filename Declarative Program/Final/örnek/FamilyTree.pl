%%%CAL�SKAN FAMILY
%Facts
male(sadula).
female(nefide).

male(ishak).
male(ismail).
female(melike).
male(erdem).
female(gulfide).

female(oya).
female(suna).
male(can).
male(okan).
male(orhan).
uncle(mumin).

male(hasan).
male(ali).
male(sakir).
%parent(X,Y):X is Y's parent.
parent(sadula,ishak).
parent(sadula,ismail).
parent(sadula,melike).
parent(sadula,erdem).
parent(sadula,gulfide).
parent(nefide,ishak).
parent(nefide,ismail).
parent(nefide,melike).
parent(nefide,erdem).
parent(nefide,gulfide).
parent(mumin,oya).
parent(mumin,suna).
parent(mumin,can).
parent(mumin,okan).
parent(mumin,orhan).
parent(ali,sadula).
parent(ali,mumin).
parent(sakir,ali).
parent(sakir,hasan).

%Rules
%father(X,Y):X is Y's father.
father(X,Y):- male(X),parent(X,Y).
%mother(X,Y):X is Y's mother.
mother(X,Y):- female(X),parent(X,Y).
%sister(X,Y):X is Y's sister.
sister(X,Y) :- female(X),parent(Par,X),parent(Par,Y), X \= Y.
%brother(X,Y):X is Y's brother.
brother(X,Y) :- male(X),parent(Par,X),parent(Par,Y), X \= Y.
%uncle(X,Y):X is Y's uncle.
uncle(X,Y):- parent(Z,Y), brother(Z,X).
%grand_uncle(X,Y):X is Y's grand uncle.
grand_uncle(X,Y):- brother(X,Z),parent(Z,A),parent(A,Y),male(X).



















