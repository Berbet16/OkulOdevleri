initial(q0).

%durumlar� belirleme
final(q1).
final(q2).
final(q3).
final(q4).
final(q5).
final(q6).
final(q7).
final(q8).
final(q9).
final(q10).


%�evirilecek ifadelerin yollar�n� belirleme
t(q0,verb,q1).
t(q1,not,q2).
t(q1,pastheard,q3).
t(q3,past,q4).
t(q2,past,q5).
t(q5,personattachment,q6).
t(q2,fusion,q7).
t(q7,future,q8).
t(q8,past,q9).
t(q9,personattachment,q10).


%fiilleri belirleme
allomorph(yap,verb).
allomorph(anla,verb).
allomorph(sev,verb).
allomorph(g�r,verb).
allomorph(git,verb).

%olumsuzluk eklerini belirleme
allomorph(ma,not).
allomorph(me,not).
allomorph(m��,pastheard).


%kayna�t�rma harfini belirleme
allomorph(y,fusion).

%ge�mi� zaman kelimelerini belirleme
allomorph(d�,past).
allomorph(di,past).
allomorph(t�,past).
allomorph(t�,past).

%gelecek zaman ve ki�ilik ekini belirleme
allomorph(acak,future).
allomorph(m,personattachment).



analyzer(String,List_of_Morphemes):-
   initial(State),
   analyzer(String,State,List_of_Morphemes).

analyzer('',State,[]):- final(State).

analyzer(String,CurrentState,[Morpheme|Morphemes]):-
   concat(Prefix,Suffix,String),
   allomorph(Prefix,Morpheme),
   t(CurrentState,Morpheme,NextState),
   analyzer(Suffix,NextState,Morphemes).











