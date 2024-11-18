int cpu_signal_handler(int host_signum, void *pinfo,

                       void *puc)

{

    siginfo_t *info = pinfo;

#if defined(__FreeBSD__) || defined(__FreeBSD_kernel__)

    ucontext_t *uc = puc;

#else

    ucontext_t *uc = puc;

#endif

    unsigned long pc;

    int is_write;



    pc = IAR_sig(uc);

    is_write = 0;

#if 0

    /* ppc 4xx case */

    if (DSISR_sig(uc) & 0x00800000) {

        is_write = 1;

    }

#else

    if (TRAP_sig(uc) != 0x400 && (DSISR_sig(uc) & 0x02000000)) {

        is_write = 1;

    }

#endif

    return handle_cpu_signal(pc, (unsigned long)info->si_addr,

                             is_write, &uc->uc_sigmask);

}
