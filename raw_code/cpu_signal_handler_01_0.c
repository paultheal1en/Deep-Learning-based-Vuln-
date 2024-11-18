int cpu_signal_handler(int host_signum, void *pinfo,

                           void *puc)

{

    siginfo_t *info = pinfo;

    ucontext_t *uc = puc;

    uint32_t *pc = uc->uc_mcontext.sc_pc;

    uint32_t insn = *pc;

    int is_write = 0;



    /* XXX: need kernel patch to get write flag faster */

    switch (insn >> 26) {

    case 0x0d: /* stw */

    case 0x0e: /* stb */

    case 0x0f: /* stq_u */

    case 0x24: /* stf */

    case 0x25: /* stg */

    case 0x26: /* sts */

    case 0x27: /* stt */

    case 0x2c: /* stl */

    case 0x2d: /* stq */

    case 0x2e: /* stl_c */

    case 0x2f: /* stq_c */

        is_write = 1;

    }



    return handle_cpu_signal(pc, (unsigned long)info->si_addr,

                             is_write, &uc->uc_sigmask);

}
