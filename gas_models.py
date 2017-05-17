EPM2_atoms = {
#FF_type  at_mass  sig[A]  eps[kcal/mol]  charge   
        "Cx":(12.0000, 2.757, 0.0559, +0.6512),
        "Ox":(15.9994, 3.033, 0.16  , -0.3256)
        }

EPM2_angles = {
        # type K theta0
        "Ox_Cx_Ox": (295.41, 180)
        }

TraPPE_atoms = {
#FF_type  at_mass  sig[A]  eps[kcal/mol]  charge   
        "Xn":(0.000001, 0.000, 0.0000, +0.964 ),
        "Nx":(14.0067, 3.310, 0.07154, -0.482 )
        "Cx"=(12.0107, 2.800, 0.05365, 0.70),
        "Ox"=(15.9994, 3.050, 0.15699, -0.35)
        }

TraPPE_angles = {
        # type K theta0
        "Nx_X_Nx": (295.41, 180)
        "Ox_Cx_Ox": (295.41, 180)
        }
