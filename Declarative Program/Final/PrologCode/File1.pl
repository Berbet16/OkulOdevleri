% this programs decipts family relationships
/*domains
     name=symbol
predicates
     parent(name,name)
     female(name)
     male(name)
     mother(name,name)
     father(name,name)
     haschild(name)
     sister(name,name)
     brother(name,name)
clauses*/


%SOYLU FAMILY!!

male(selahattin).
female(gulseren).

male(sebahattin).
male(ferdi).
female(melike).
female(ayse).
female(betulberna).

female(ipek).
female(saliha).
male(ali).
male(mehmet).
male(osman).
uncle(huseyin).

male(kemal).
male(ahmet).
male(dursun).


%parent(X,Y): Xis Y's parent!!!
parent(selahattin,sebahattin).
parent(selahattin,ferdi).
parent(selahattin,melike).
parent(selahattin,ayse).
parent(selahattin,betulberna).

parent(gulseren,sebahattin).
parent(gulseren,ferdi).
parent(gulseren,melike).
parent(gulseren,ayse).
parent(gulseren,betulberna).

parent(huseyin,ipek).
parent(huseyin,saliha).
parent(huseyin,ali).
parent(huseyin,mehmet).
parent(huseyin,osman).

parent(ahmet,gulseren).
parent(ahmet,huseyin).

parent(dursun,ahmet).
parent(dursun,kemal).

%RULESS!!

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







