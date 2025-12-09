Install HighFold https://github.com/hongliangduan/HighFold 
In the 1-predict-cyclic.py, there are three parameters needed to be set up 
  1. core (what is the core peptide sequence?)
  2. span length (how many residues do you want to expand?)
  3. number of sample (how many output sequences do you want?)
run 1-predict-cyclic.py gives you the predict.fasta in the ./output folder
run 2-run-highfold.sh gives you the highfold modeling results in the ./output folder
run 3-final gives you a output.csv which contains the cyclic peptide sequences, pLDDT scores, and other properties

Note: there are some restrictions
1. the total cyclic peptide sequence cannot be too long. Most training data are <15 aa. Don't predict >20 aa cyclic peptide
2. If the core is short, don't predict too long. For example, the core length = 1, and span length = 10. This is not practical
   A good case is, for example, core length = 3, and span length = 7, make a total of 10 aa cyclic peptide. That is to say core length cannot be < 30% of the total length.
