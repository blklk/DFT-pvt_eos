# pvt_eos
Quick PVT EOS fitter / evaluator for project student

To run:

1. Make input files (see the test directory)
2. In the fdirectory with the data files run .
   
   python DFT-pvt.py feal25-thermo_calc/*_gpa/bridgmanite.castep --polyplot feal25-4-PLOYPLOT.png --plot_both feal25-4-BOTH.png
   
   python DFT-pvt-kp04.py feal25-thermo_calc/*_gpa/bridgmanite.castep --fixed_Kp0 4 --polyplot feal25-4-PLOYPLOT.png --plot_both feal25-4-BOTH.png

    # Old, many work though: ../bm3_eos.py --plot_both both_plot.eps --polyplot polyplot.eps ??GPa.txt ???GPa.txt
