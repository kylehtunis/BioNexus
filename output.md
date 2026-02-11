No filename given, using 'Kinetics.csv'

============================================================
MECHANISM IDENTIFICATION (Sub-question 1)
============================================================

--- Ping-Pong Bi-Bi Model ---
  Vmax = 1.0000
  Km1 (S1) = 0.1000
  Km2 (S2) = 0.1000
  R² = 1.000000
  Chi² = 0.000000

--- Sequential (Ternary Complex) Model ---
  Vmax = 1.0000
  Km1 (S1) = 0.1000
  Km2 (S2) = 0.1000
  Ks1 = -0.000000
  R² = 1.000000
  Chi² = 0.000000

--- Lineweaver-Burk Slope Analysis ---
  S2           LB Slope       
  ---------------------------
  0.05         0.100000       
  0.1          0.100000       
  0.2          0.100000       
  0.5          0.100000       
  1            0.100000       
  1e+04        0.100000       
  Slope CV = 0.0000
  -> Slopes are approximately parallel (consistent with Ping-Pong)

  CONCLUSION: Europase follows a Ping-Pong Bi-Bi mechanism.
============================================================

Analyzing multiple S2 conditions:

S2           Km           Vmax         R²          
------------------------------------------------
0.05         0.0333       0.3333       1.0000      
0.1          0.0500       0.5000       1.0000      
0.2          0.0667       0.6667       1.0000      
0.5          0.0833       0.8333       1.0000      
1            0.0909       0.9091       1.0000      
1e+04        0.1000       1.0000       1.0000      

============================================================
BATCH REACTOR TIME (Sub-question 3)
============================================================
  S1 initial: 100 g/L = 666.67 mM
  S1 final:   1 g/L   = 6.67 mM
  Km (S2 excess) = 0.1000 mM
  Vmax (S2 excess) = 1.0000 mM/s

  Time = (Km * ln(S0/Sf) + (S0 - Sf)) / Vmax
  Time = 660.47 seconds
       = 11.01 minutes
============================================================
