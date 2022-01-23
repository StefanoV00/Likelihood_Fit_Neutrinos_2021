There are 5 documents. All print major results and produce relevant plots.
Note: all documents but last Testing also use extract to get the data. The
argument of extract must be changed accordingly (it is the path of the data
file).

The documents are:

- Project_function: contains all relevant functions, needs to be imported
by all others.

- Project_Part3: contains code required to answer tasks in section3 of 
script and produce material for sections I, II, III of report.
NOTE: the contour plot of the NLL was taken with 1,000,000 total
evaluations, 1000 values of theta, 1000 of Deltam^2; in lines 116 and 117
those have been substituted by 100 and 100 for a total of 10,000 evaluations,
to allow production of plot in reasonable amount of time. Feel free
to switch them back to 1000 if required. 

- Project_Part4: code to answer to tasks in section4 and produce material
for section IV of report (testing of methods apart).

- Project_Part5: code to answer to tasks in section5 and sproduce material 
for section V of report. Note: it takes (at least to me) 3-4 minutes,
mainly due to implementation of Metropolis and Quasi-Netwon methods.

- Project_Testing: code to produce testing of functions, Fig 4.2, Table 4.1 
of report. Takes time. line 406: N has been changed from the original 100 to
10, to allow computation in more reasonable amount time. Feel fry to change
it to original value. 