for i in {1..4}
do
for j in {1..4}
do
   echo "\section{}" >> ${i}-${j}.tex
done
done

for i in {5..6}
do
   mv  0${i}.tex Z0${i}.tex
done

for i in {0..6}
do
   rm  0${i}.aux 
done
for i in {10..20}
