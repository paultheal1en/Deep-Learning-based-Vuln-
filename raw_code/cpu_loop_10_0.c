void cpu_loop(CPUS390XState *env)

{

    CPUState *cs = CPU(s390_env_get_cpu(env));

    int trapnr, n, sig;

    target_siginfo_t info;

    target_ulong addr;



    while (1) {

        cpu_exec_start(cs);

        trapnr = cpu_s390x_exec(cs);

        cpu_exec_end(cs);

        switch (trapnr) {

        case EXCP_INTERRUPT:

            /* Just indicate that signals should be handled asap.  */

            break;



        case EXCP_SVC:

            n = env->int_svc_code;

            if (!n) {

                /* syscalls > 255 */

                n = env->regs[1];

            }

            env->psw.addr += env->int_svc_ilen;

            env->regs[2] = do_syscall(env, n, env->regs[2], env->regs[3],

                                      env->regs[4], env->regs[5],

                                      env->regs[6], env->regs[7], 0, 0);

            break;



        case EXCP_DEBUG:

            sig = gdb_handlesig(cs, TARGET_SIGTRAP);

            if (sig) {

                n = TARGET_TRAP_BRKPT;

                goto do_signal_pc;

            }

            break;

        case EXCP_PGM:

            n = env->int_pgm_code;

            switch (n) {

            case PGM_OPERATION:

            case PGM_PRIVILEGED:

                sig = TARGET_SIGILL;

                n = TARGET_ILL_ILLOPC;

                goto do_signal_pc;

            case PGM_PROTECTION:

            case PGM_ADDRESSING:

                sig = TARGET_SIGSEGV;

                /* XXX: check env->error_code */

                n = TARGET_SEGV_MAPERR;

                addr = env->__excp_addr;

                goto do_signal;

            case PGM_EXECUTE:

            case PGM_SPECIFICATION:

            case PGM_SPECIAL_OP:

            case PGM_OPERAND:

            do_sigill_opn:

                sig = TARGET_SIGILL;

                n = TARGET_ILL_ILLOPN;

                goto do_signal_pc;



            case PGM_FIXPT_OVERFLOW:

                sig = TARGET_SIGFPE;

                n = TARGET_FPE_INTOVF;

                goto do_signal_pc;

            case PGM_FIXPT_DIVIDE:

                sig = TARGET_SIGFPE;

                n = TARGET_FPE_INTDIV;

                goto do_signal_pc;



            case PGM_DATA:

                n = (env->fpc >> 8) & 0xff;

                if (n == 0xff) {

                    /* compare-and-trap */

                    goto do_sigill_opn;

                } else {

                    /* An IEEE exception, simulated or otherwise.  */

                    if (n & 0x80) {

                        n = TARGET_FPE_FLTINV;

                    } else if (n & 0x40) {

                        n = TARGET_FPE_FLTDIV;

                    } else if (n & 0x20) {

                        n = TARGET_FPE_FLTOVF;

                    } else if (n & 0x10) {

                        n = TARGET_FPE_FLTUND;

                    } else if (n & 0x08) {

                        n = TARGET_FPE_FLTRES;

                    } else {

                        /* ??? Quantum exception; BFP, DFP error.  */

                        goto do_sigill_opn;

                    }

                    sig = TARGET_SIGFPE;

                    goto do_signal_pc;

                }



            default:

                fprintf(stderr, "Unhandled program exception: %#x\n", n);

                cpu_dump_state(cs, stderr, fprintf, 0);

                exit(EXIT_FAILURE);

            }

            break;



        do_signal_pc:

            addr = env->psw.addr;

        do_signal:

            info.si_signo = sig;

            info.si_errno = 0;

            info.si_code = n;

            info._sifields._sigfault._addr = addr;

            queue_signal(env, info.si_signo, &info);

            break;



        default:

            fprintf(stderr, "Unhandled trap: 0x%x\n", trapnr);

            cpu_dump_state(cs, stderr, fprintf, 0);

            exit(EXIT_FAILURE);

        }

        process_pending_signals (env);

    }

}
