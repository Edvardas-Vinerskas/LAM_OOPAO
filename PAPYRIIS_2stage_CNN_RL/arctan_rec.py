from ozitelemetry.OZITelemetry import OZITele



tele = OZITele(tele_path = 'your_path.npy', is_onsky = True)
tele.reconstruct_all_phase(mehtod = 'atan', iteration = 10, damping = 0.5)


tele.phase
tele.OPDs