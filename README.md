# Term Project for ECE 236c: Comaring 2nd order optimizers

## Todo
Overall: 
- [ ] Figure out which optimizations from [Martens and Sutskever(2012) section 20](https://link.springer.com/chapter/10.1007/978-3-642-35289-8_27) are applicable to LM.
     - [ ] linesearch
     - [ ] backtracking
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
  

Yiming:
-[ ] Coding
  - [ ] Add CG adaptive momentum to the LM code
  - [ ] Add above determined optimizations such as linesearch, backtracking, trust reigons etc. 


Omead:
- [ ] Coding
  - [x] Add KFAC/EKFAC
  - [ ] Add Hessian Free optimization to the comparisons: https://github.com/fmeirinhos/pytorch-hessianfree 
  - [ ] Add Timing analysis 
- [ ] Have Louis give the updated classificaiton code.
- [ ] Make summary of MST paper


![GitHub Logo](/algorithmComparison.png)
