# Term Project for ECE 236c: Comaring 2nd order optimizers

## Todo
Overall: 
- [ ] Figure out which optimizations from [Martens and Sutskever(2012) section 20](https://link.springer.com/chapter/10.1007/978-3-642-35289-8_27) are applicable to LM.
     - [ ] ~~linesearch~~
     - [ ] ~~backtracking~~
     - [ ] trust reigons
     - [ ] etc  
- [ ] Coding
  - [ ] Speed up overall code 
  - [ ] Fix issue of for loop in Hessian sampleing 
  - [ ] Increase network size 
  - [ ] Improve nework with Conv Layers etc.
  - [ ] Add C-MST code/rerun tests 

- [ ] Writeup(Week 10)
  - [ ] Break down math of methods we use for writeup 
  - [ ] What are the tradeoffs 
  - [ ] Show on different datasets
  - [ ] Show different tasks
 
 - [ ] Questions: 
      - [ ] Can we compare Hessian-Free and LBFGS since they do multiple iterations per-teration?
      - [ ] Do we count failed iterations as iterations?
      - [ ] How can we compare with the multiple iterations per step in HF and LBFGS
  

Yiming:
- [ ] Coding
  - [ ] Add CG adaptive momentum to the LM code
  - [ ] Add above determined optimizations such as linesearch, backtracking, trust reigons etc. 
  - [ ] Look at various adaptive trust reigon methods


Omead:
- [ ] Coding
  - [x] Add KFAC/EKFAC
  - [x] Add [Hessian Free]( https://github.com/fmeirinhos/pytorch-hessianfree) optimization to the comparisons 
  - [x] Add L-BFGS
  - [ ] Add Timing analysis 
  - [x] ~~Add Learning rate Search with trust reigion~~
- [ ] Have Louis give the updated classificaiton code.
- [ ] Make summary of MST paper



## Include Failed Iterations of LM
![GitHub Logo](/algorithmComparison_counting_failed_iterations.png)

## Search over LR of LM
![GitHub Logo](/algorithmComparison_with_lr_search.png)

## Without Failed Iterations of LM
![GitHub Logo](/algorithmComparison_no_ls.png)


