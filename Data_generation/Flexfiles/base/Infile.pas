turb
**** Blade data ****
IEA10MW.VDA                    Blade data filename,  '-'= data below
**** Rotor + Nacelle ***
3                            B  (number of blades)
1.0 1.0  1.0 1.0  1.0 1.0    K_flap_fak  K_edge_fak, pairs, blade 1..3 ||Ændres hvis man vil ændre stivhed af vinger
1.0 1.0 1.0	             M_fak, blade 1..3  ||Ændres hvis man vil ændre stivhed af vinger
0  0  0                      Pitch offset blade 1..B (deg)
0                            X_rod (offset blade data x) (m) || Hvis første punkt på vinge har radius=0, selvom det er en distance (x_rod) fra centrum
4.0   6.0                    coning  tilt  (deg, deg)
7.1  2.96                  Znav  Zrn    (rotor overhang + pos. of shaft bend) 7.1 in report page 36, but 10.039 in OpenFast?
81707  0             	     Mnav  Zgnav
4.76512E5 4.84477E5  4.84272E5    Ixnav  Iynav  Iznav, see HAWC2
1.3E10 1.3E10 2.05E9       ? Kgax  Kgay  Ktors  (shaft stiffness, DOF 14,15,28)  ||Scale 2.67 i forhold til NREL5MW
1330.1  50.0      	     Igenerator  Ngear (3801700-476512)/Ngear^2. GenIner = 3801700, HubIner = 476512, see HAWC2 file
545623.0  2.69  2.75         Mkab  Zgkab  XKK2 545623.0  446034.0 see IEA report p. 36. 
6.54637E5  6.37319E5  3.15523E5       Ixkab  Iykab  Izkab
3.6E9  1.0E10                Ktx  KKy  (yaw- and tilt stiffness) 
30  20   0.0                 CdARz CdARxy ZlatR  aero drag of nav ||Cd*A i z,x,y retning og distance fra punkt R til krafts angrebspunkt, kun vigtigt i storm
30  105  2.0         	     CdAKz CdAKxy ZlatK  aero drag of nacelle ||Cd*A i z,x,y retning og distance fra punkt K til krafts angrebspunkt, kun vigtigt i storm
.03 .03 .03 .03 .03     	     Damp. DOF 11,12 + 14,15 + shaft tors. (log.decr.)
****  tower data ****
IEA10MW.TDA            Tower data filename,  '-'= data below
****  foundation data ****
Fund_v02_h0.fda              Foundation data filename,  '-'= data below
**** operational data  ****
1.225  9.81           Ro  g
omegachange  pitchchange  1  1         Omega  Tetap  Generator-on/off ((1 = on, 0 = off)
180  yawchange                Psi   Yaw  (rotorpos  yawpos, deg,deg)
10.0  0.0  0  0      Vnav  Vexp  Vdir  Vslope
1.0  1.0  1.0        Turb.intens (u), Rel.ti. (v) and (w) 0.12  0.8  0.5
inflow.flxU 0    Turbulence-filename  T-offset  (u)
inflow.flxV 0   Turbulence-filename  T-offset  (v)
inflow.flxW 0   Turbulence-filename  T-offset  (w)
**** data for simulation ***
1 1 1 0             	Blade-dof: 1F 2F 1K 2K,   1 = active, 0=stiff
1 1 1 1 1 1          	DOF 11..15 + 28 (shafttors) 1=active, 0=stiff, DOF11=yaw
1 1 1 1              	DOF 7..10 (twr: L1, L2, T1, T2)  L = long., T=transv.   
0 0 0 0 0 0           	DOF 1..6 (foundat: Tx, Tz,Ry, Ty,Rz, Rx) (T=transl,R=rot)
0 0.02 700.0       	Tstart  dt  Tmax 19360
0  20   2   2        	Printop  Nprint  Filop  Nfil
1 0.11 0.05 40 10 4  	Stall dClda dCldaS AlfS ALrund TauFak
0.0    1.0             	Dynamisk wake (0/1)  DynWTCfak
0.0  1.0             	Towershadow-factor  Cd-factor (twr)
***  data for generator, brake, pitch etc. ****
-Gendat.inf             Generatordata filename, '-'= data below
-Brakedat.inf           Brakedata filename,     '-'= data below
-Pitchdat.inf           Pitchdata filename,     '-'= data below
-Yawdat.inf             Yaw data filename,      '-'= data below
-Contrdat,inf           Controldata filename,   '-'= data belov
0-Winddat.inf         	Winddata filename,      '-'= data below, '0'=no data
-Initdat.inf            Initdata filename,      '-'= default values used
-Restart.rstF          Restart filename,       '-'= default values used   -Restart.rstF
*** Inputdata for Flex5 generator version 2.1 ***
DTU10MWVS 
10000  0.05  0.2        Pref (kW), tau_g (s)  tau_set (s)
5.0  0.9             	F0 (Hz), Ksi (-), 2.order bandpass RPM-filter 5.0 0.7 see *htc line 535
1000                     D_gen (Nm/rpm)  Generator damping  
180  296.61   593.22    P_loss (kW) at 0%, 50%, 100% Pref see HAWC2toFlex5_PitchTable.m
14 480                	P_loss_mech Nref (mech. loss (kW) at RPM, gen. off) 14 480
*** Inputdata for Flex5 brake model version 2.0 ***
IEA10MW 
52.3                	Dynamic brake moment on generator shaft (kNm)
62.8                 	Static brake moment (kNm)
0.7                   	tau      (s)
0.0                    	T-delay  (s)
0.0  0.25  0.05        	V (deg), R/V, K0/Ktors  ( mainshaft play )
*** Inputdata for Flex5 pitch model version 1.2 ***
IEA 10MW pitch servo  : "Blade pitch servo and generator models are not included in this controller and should be modeled separately, if they are to be included in the simulations."
6.3  0.7         	OMres (rad/s)  Ksi-rel (< 1) OMres=resonance frequency (rad/s) 6.3 0.7, see *htc line 590
*** Inputdata for Flex5 yaw model version 1.0 ***
IEA 10MW
1.0  0.2               	Yaw-rate (deg/s)  Yaw-tau (s)
*** Inputdata for Flex5 control system version 3.2 ***
IEA 10MW, variable speed
390  6123      	N1, Pow1  (rpm, kW) point on cubic part of P(rpm) curve see HAWC2toFlecx5_PitchTable.m 347.0550 4099.4
300 433.8              	N_min, N_max (rpm, rpm)
10000                	Pmax (kW)
25 250                  KI KP (Nm/s/rpm, Nm/rpm) const RPM control 25 250
0.2 0.7                 F0 (Hz), Ksi (-)  2.order RPM-filter    0.2 0.7
0.0303  0.0666  0.0     KI KP KD (deg/s/rpm, deg/rpm, deg/rpm*s) 0.0303 0.0666 0
6.07                    KK (deg) (reduction of gain as function of pitch) 6.0
90  10           	Teta_max Pitchrate_max  (deg,deg/s)
5                   	TauV (sec)  (wind-averaging for minimum pitch)
9    No. of lines in table for Wind, minPitch, pow(init) (m/s,deg, kW)
4    4.0493   388
5    3.1213   937
6    1.7476  1712
7   -0.250     2747
8   -0.250     4099
9   -0.250     5838
10  -0.44     8013
10.5  -1.50   9300
11   0.0   10000
-0.2  2  10          TePstart TePstop TePsstop (deg/s)
0.05                 Tsamp  (sec)   control system
*** Inputdata for deterministic wind, version 1.1 ***
NM80 test 
94                   N  number of lines in interpolation table below
40      5        0           T  Vnav  Vdir    (s, m/s, deg)          
          41           5           0
         460           5           0
         461           5           0
         880           5           0
         881           6           0
        1300           6           0
        1301           7           0
        1720           7           0
        1721           8           0
        2140           8           0
        2141           9           0
        2560           9           0
        2561          10           0
        2980          10           0
        2981          11           0
        3400          11           0
        3401          12           0
        3820          12           0
        3821          13           0
        4240          13           0
        4241          14           0
        4660          14           0
        4661          15           0
        5080          15           0
        5081          16           0
        5500          16           0
        5501          17           0
        5920          17           0
        5921          18           0
        6340          18           0
        6341          19           0
        6760          19           0
        6761          20           0
        7180          20           0
        7181          21           0
        7600          21           0
        7601          22           0
        8020          22           0
        8021          23           0
        8440          23           0
        8441          24           0
        8860          24           0
        8861          25           0
        9280          25           0
        9281          25           0
        9700          25           0
        9701          25           0
       10120          25           0
       10121          24           0
       10540          24           0
       10541          23           0
       10960          23           0
       10961          22           0
       11380          22           0
       11381          21           0
       11800          21           0
       11801          20           0
       12220          20           0
       12221          19           0
       12640          19           0
       12641          18           0
       13060          18           0
       13061          17           0
       13480          17           0
       13481          16           0
       13900          16           0
       13901          15           0
       14320          15           0
       14321          14           0
       14740          14           0
       14741          13           0
       15160          13           0
       15161          12           0
       15580          12           0
       15581          11           0
       16000          11           0
       16001          10           0
       16420          10           0
       16421           9           0
       16840           9           0
       16841           8           0
       17260           8           0
       17261           7           0
       17680           7           0
       17681           6           0
       18100           6           0
       18101           5           0
       18520           5           0
       18521           5           0
       18940           5           0
       18941           5           0
       19360           5           0
       19361           5           0

