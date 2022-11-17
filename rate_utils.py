def lor(e,gamma,e0):
    return gamma / ( (e-e0)*(e-e0) + gamma*(gamma/4)  )

def bose_einstein(e,beta):
    return 1.0 / (np.exp(beta*e) - 1)

def fermi_dirac(e,mu,beta):
    return 1.0 / ( np.exp(beta*(e-mu)) + 1 )

def transfer_rate(e,mu,gamma_i,gamma_f,e_i,e_f,beta,kappa,w):
    print('Evaluating Lorentzians...')
    lor_start = perf_counter()
    lor_i = lor(e,gamma_i,e_i) 
    lor_f = lor(e+w,gamma_f,e_f)
    lor_end = perf_counter()
    print('Done with Lorentzians. Computation time [seconds] = ', lor_end - lor_start)


    print('Evaluating Fermi Dirac distributions...')
    fd_start = perf_counter()
    fd_i = fermi_dirac(e,mu,beta)
    cfd_f = 1 - fermi_dirac(e+w,-mu,beta)
    fd_end = perf_counter()
    print('Done with Fermi Dirac distributions. Computation time [seconds] = ', fd_end - fd_start)

    prefactor = kappa*kappa
    print(prefactor.shape)
    print('Evaluating integral...')
    int_start = perf_counter()
    integral = simpson(lor_i*lor_f*fd_i*cfd_f, e, axis=-1)
    int_end = perf_counter()
    print('Done with integral. Computation time [seconds] = ', int_end - int_start)
    out = prefactor[:,None,None]*integral

    print(out.shape)

    return out / (2.0*np.pi)