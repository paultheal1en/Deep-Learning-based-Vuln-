void cpu_loop(CPUPPCState *env)

{

    target_siginfo_t info;

    int trapnr;

    uint32_t ret;



    for(;;) {

        trapnr = cpu_ppc_exec(env);

        switch(trapnr) {

        case POWERPC_EXCP_NONE:

            /* Just go on */

            break;

        case POWERPC_EXCP_CRITICAL: /* Critical input                        */

            cpu_abort(env, "Critical interrupt while in user mode. "

                      "Aborting\n");

            break;

        case POWERPC_EXCP_MCHECK:   /* Machine check exception               */

            cpu_abort(env, "Machine check exception while in user mode. "

                      "Aborting\n");

            break;

        case POWERPC_EXCP_DSI:      /* Data storage exception                */

            EXCP_DUMP(env, "Invalid data memory access: 0x" ADDRX "\n",

                      env->spr[SPR_DAR]);

            /* XXX: check this. Seems bugged */

            switch (env->error_code & 0xFF000000) {

            case 0x40000000:

                info.si_signo = TARGET_SIGSEGV;

                info.si_errno = 0;

                info.si_code = TARGET_SEGV_MAPERR;

                break;

            case 0x04000000:

                info.si_signo = TARGET_SIGILL;

                info.si_errno = 0;

                info.si_code = TARGET_ILL_ILLADR;

                break;

            case 0x08000000:

                info.si_signo = TARGET_SIGSEGV;

                info.si_errno = 0;

                info.si_code = TARGET_SEGV_ACCERR;

                break;

            default:

                /* Let's send a regular segfault... */

                EXCP_DUMP(env, "Invalid segfault errno (%02x)\n",

                          env->error_code);

                info.si_signo = TARGET_SIGSEGV;

                info.si_errno = 0;

                info.si_code = TARGET_SEGV_MAPERR;

                break;

            }

            info._sifields._sigfault._addr = env->nip;

            queue_signal(info.si_signo, &info);

            break;

        case POWERPC_EXCP_ISI:      /* Instruction storage exception         */

            EXCP_DUMP(env, "Invalid instruction fetch: 0x\n" ADDRX "\n",

                      env->spr[SPR_SRR0]);

            /* XXX: check this */

            switch (env->error_code & 0xFF000000) {

            case 0x40000000:

                info.si_signo = TARGET_SIGSEGV;

            info.si_errno = 0;

                info.si_code = TARGET_SEGV_MAPERR;

                break;

            case 0x10000000:

            case 0x08000000:

                info.si_signo = TARGET_SIGSEGV;

                info.si_errno = 0;

                info.si_code = TARGET_SEGV_ACCERR;

                break;

            default:

                /* Let's send a regular segfault... */

                EXCP_DUMP(env, "Invalid segfault errno (%02x)\n",

                          env->error_code);

                info.si_signo = TARGET_SIGSEGV;

                info.si_errno = 0;

                info.si_code = TARGET_SEGV_MAPERR;

                break;

            }

            info._sifields._sigfault._addr = env->nip - 4;

            queue_signal(info.si_signo, &info);

            break;

        case POWERPC_EXCP_EXTERNAL: /* External input                        */

            cpu_abort(env, "External interrupt while in user mode. "

                      "Aborting\n");

            break;

        case POWERPC_EXCP_ALIGN:    /* Alignment exception                   */

            EXCP_DUMP(env, "Unaligned memory access\n");

            /* XXX: check this */

            info.si_signo = TARGET_SIGBUS;

            info.si_errno = 0;

            info.si_code = TARGET_BUS_ADRALN;

            info._sifields._sigfault._addr = env->nip - 4;

            queue_signal(info.si_signo, &info);

            break;

        case POWERPC_EXCP_PROGRAM:  /* Program exception                     */

            /* XXX: check this */

            switch (env->error_code & ~0xF) {

            case POWERPC_EXCP_FP:

                EXCP_DUMP(env, "Floating point program exception\n");

                info.si_signo = TARGET_SIGFPE;

                info.si_errno = 0;

                switch (env->error_code & 0xF) {

                case POWERPC_EXCP_FP_OX:

                    info.si_code = TARGET_FPE_FLTOVF;

                    break;

                case POWERPC_EXCP_FP_UX:

                    info.si_code = TARGET_FPE_FLTUND;

                    break;

                case POWERPC_EXCP_FP_ZX:

                case POWERPC_EXCP_FP_VXZDZ:

                    info.si_code = TARGET_FPE_FLTDIV;

                    break;

                case POWERPC_EXCP_FP_XX:

                    info.si_code = TARGET_FPE_FLTRES;

                    break;

                case POWERPC_EXCP_FP_VXSOFT:

                    info.si_code = TARGET_FPE_FLTINV;

                    break;

                case POWERPC_EXCP_FP_VXSNAN:

                case POWERPC_EXCP_FP_VXISI:

                case POWERPC_EXCP_FP_VXIDI:

                case POWERPC_EXCP_FP_VXIMZ:

                case POWERPC_EXCP_FP_VXVC:

                case POWERPC_EXCP_FP_VXSQRT:

                case POWERPC_EXCP_FP_VXCVI:

                    info.si_code = TARGET_FPE_FLTSUB;

                    break;

                default:

                    EXCP_DUMP(env, "Unknown floating point exception (%02x)\n",

                              env->error_code);

                    break;

                }

                break;

            case POWERPC_EXCP_INVAL:

                EXCP_DUMP(env, "Invalid instruction\n");

                info.si_signo = TARGET_SIGILL;

                info.si_errno = 0;

                switch (env->error_code & 0xF) {

                case POWERPC_EXCP_INVAL_INVAL:

                    info.si_code = TARGET_ILL_ILLOPC;

                    break;

                case POWERPC_EXCP_INVAL_LSWX:

                    info.si_code = TARGET_ILL_ILLOPN;

                    break;

                case POWERPC_EXCP_INVAL_SPR:

                    info.si_code = TARGET_ILL_PRVREG;

                    break;

                case POWERPC_EXCP_INVAL_FP:

                    info.si_code = TARGET_ILL_COPROC;

                    break;

                default:

                    EXCP_DUMP(env, "Unknown invalid operation (%02x)\n",

                              env->error_code & 0xF);

                    info.si_code = TARGET_ILL_ILLADR;

                    break;

                }

                break;

            case POWERPC_EXCP_PRIV:

                EXCP_DUMP(env, "Privilege violation\n");

                info.si_signo = TARGET_SIGILL;

                info.si_errno = 0;

                switch (env->error_code & 0xF) {

                case POWERPC_EXCP_PRIV_OPC:

                    info.si_code = TARGET_ILL_PRVOPC;

                    break;

                case POWERPC_EXCP_PRIV_REG:

                    info.si_code = TARGET_ILL_PRVREG;

                    break;

                default:

                    EXCP_DUMP(env, "Unknown privilege violation (%02x)\n",

                              env->error_code & 0xF);

                    info.si_code = TARGET_ILL_PRVOPC;

                    break;

                }

                break;

            case POWERPC_EXCP_TRAP:

                cpu_abort(env, "Tried to call a TRAP\n");

                break;

            default:

                /* Should not happen ! */

                cpu_abort(env, "Unknown program exception (%02x)\n",

                          env->error_code);

                break;

            }

            info._sifields._sigfault._addr = env->nip - 4;

            queue_signal(info.si_signo, &info);

            break;

        case POWERPC_EXCP_FPU:      /* Floating-point unavailable exception  */

            EXCP_DUMP(env, "No floating point allowed\n");

            info.si_signo = TARGET_SIGILL;

            info.si_errno = 0;

            info.si_code = TARGET_ILL_COPROC;

            info._sifields._sigfault._addr = env->nip - 4;

            queue_signal(info.si_signo, &info);

            break;

        case POWERPC_EXCP_SYSCALL:  /* System call exception                 */

            cpu_abort(env, "Syscall exception while in user mode. "

                      "Aborting\n");

            break;

        case POWERPC_EXCP_APU:      /* Auxiliary processor unavailable       */

            EXCP_DUMP(env, "No APU instruction allowed\n");

            info.si_signo = TARGET_SIGILL;

            info.si_errno = 0;

            info.si_code = TARGET_ILL_COPROC;

            info._sifields._sigfault._addr = env->nip - 4;

            queue_signal(info.si_signo, &info);

            break;

        case POWERPC_EXCP_DECR:     /* Decrementer exception                 */

            cpu_abort(env, "Decrementer interrupt while in user mode. "

                      "Aborting\n");

            break;

        case POWERPC_EXCP_FIT:      /* Fixed-interval timer interrupt        */

            cpu_abort(env, "Fix interval timer interrupt while in user mode. "

                      "Aborting\n");

            break;

        case POWERPC_EXCP_WDT:      /* Watchdog timer interrupt              */

            cpu_abort(env, "Watchdog timer interrupt while in user mode. "

                      "Aborting\n");

            break;

        case POWERPC_EXCP_DTLB:     /* Data TLB error                        */

            cpu_abort(env, "Data TLB exception while in user mode. "

                      "Aborting\n");

            break;

        case POWERPC_EXCP_ITLB:     /* Instruction TLB error                 */

            cpu_abort(env, "Instruction TLB exception while in user mode. "

                      "Aborting\n");

            break;

        case POWERPC_EXCP_DEBUG:    /* Debug interrupt                       */

            /* XXX: check this */

            {

                int sig;



                sig = gdb_handlesig(env, TARGET_SIGTRAP);

                if (sig) {

                    info.si_signo = sig;

                    info.si_errno = 0;

                    info.si_code = TARGET_TRAP_BRKPT;

                    queue_signal(info.si_signo, &info);

                  }

            }

            break;

        case POWERPC_EXCP_SPEU:     /* SPE/embedded floating-point unavail.  */

            EXCP_DUMP(env, "No SPE/floating-point instruction allowed\n");

            info.si_signo = TARGET_SIGILL;

            info.si_errno = 0;

            info.si_code = TARGET_ILL_COPROC;

            info._sifields._sigfault._addr = env->nip - 4;

            queue_signal(info.si_signo, &info);

            break;

        case POWERPC_EXCP_EFPDI:    /* Embedded floating-point data IRQ      */

            cpu_abort(env, "Embedded floating-point data IRQ not handled\n");

            break;

        case POWERPC_EXCP_EFPRI:    /* Embedded floating-point round IRQ     */

            cpu_abort(env, "Embedded floating-point round IRQ not handled\n");

            break;

        case POWERPC_EXCP_EPERFM:   /* Embedded performance monitor IRQ      */

            cpu_abort(env, "Performance monitor exception not handled\n");

            break;

        case POWERPC_EXCP_DOORI:    /* Embedded doorbell interrupt           */

            cpu_abort(env, "Doorbell interrupt while in user mode. "

                       "Aborting\n");

            break;

        case POWERPC_EXCP_DOORCI:   /* Embedded doorbell critical interrupt  */

            cpu_abort(env, "Doorbell critical interrupt while in user mode. "

                      "Aborting\n");

            break;

        case POWERPC_EXCP_RESET:    /* System reset exception                */

            cpu_abort(env, "Reset interrupt while in user mode. "

                      "Aborting\n");

            break;

#if defined(TARGET_PPC64) && !defined(TARGET_ABI32) /* PowerPC 64 */

        case POWERPC_EXCP_DSEG:     /* Data segment exception                */

            cpu_abort(env, "Data segment exception while in user mode. "

                      "Aborting\n");

            break;

        case POWERPC_EXCP_ISEG:     /* Instruction segment exception         */

            cpu_abort(env, "Instruction segment exception "

                      "while in user mode. Aborting\n");

            break;

#endif /* defined(TARGET_PPC64) && !defined(TARGET_ABI32) */

#if defined(TARGET_PPC64H) && !defined(TARGET_ABI32)

        /* PowerPC 64 with hypervisor mode support */

        case POWERPC_EXCP_HDECR:    /* Hypervisor decrementer exception      */

            cpu_abort(env, "Hypervisor decrementer interrupt "

                      "while in user mode. Aborting\n");

            break;

#endif /* defined(TARGET_PPC64H) && !defined(TARGET_ABI32) */

        case POWERPC_EXCP_TRACE:    /* Trace exception                       */

            /* Nothing to do:

             * we use this exception to emulate step-by-step execution mode.

             */

            break;

#if defined(TARGET_PPC64H) && !defined(TARGET_ABI32)

        /* PowerPC 64 with hypervisor mode support */

        case POWERPC_EXCP_HDSI:     /* Hypervisor data storage exception     */

            cpu_abort(env, "Hypervisor data storage exception "

                      "while in user mode. Aborting\n");

            break;

        case POWERPC_EXCP_HISI:     /* Hypervisor instruction storage excp   */

            cpu_abort(env, "Hypervisor instruction storage exception "

                      "while in user mode. Aborting\n");

            break;

        case POWERPC_EXCP_HDSEG:    /* Hypervisor data segment exception     */

            cpu_abort(env, "Hypervisor data segment exception "

                      "while in user mode. Aborting\n");

            break;

        case POWERPC_EXCP_HISEG:    /* Hypervisor instruction segment excp   */

            cpu_abort(env, "Hypervisor instruction segment exception "

                      "while in user mode. Aborting\n");

            break;

#endif /* defined(TARGET_PPC64H) && !defined(TARGET_ABI32) */

        case POWERPC_EXCP_VPU:      /* Vector unavailable exception          */

            EXCP_DUMP(env, "No Altivec instructions allowed\n");

            info.si_signo = TARGET_SIGILL;

            info.si_errno = 0;

            info.si_code = TARGET_ILL_COPROC;

            info._sifields._sigfault._addr = env->nip - 4;

            queue_signal(info.si_signo, &info);

            break;

        case POWERPC_EXCP_PIT:      /* Programmable interval timer IRQ       */

            cpu_abort(env, "Programable interval timer interrupt "

                      "while in user mode. Aborting\n");

            break;

        case POWERPC_EXCP_IO:       /* IO error exception                    */

            cpu_abort(env, "IO error exception while in user mode. "

                      "Aborting\n");

            break;

        case POWERPC_EXCP_RUNM:     /* Run mode exception                    */

            cpu_abort(env, "Run mode exception while in user mode. "

                      "Aborting\n");

            break;

        case POWERPC_EXCP_EMUL:     /* Emulation trap exception              */

            cpu_abort(env, "Emulation trap exception not handled\n");

            break;

        case POWERPC_EXCP_IFTLB:    /* Instruction fetch TLB error           */

            cpu_abort(env, "Instruction fetch TLB exception "

                      "while in user-mode. Aborting");

            break;

        case POWERPC_EXCP_DLTLB:    /* Data load TLB miss                    */

            cpu_abort(env, "Data load TLB exception while in user-mode. "

                      "Aborting");

            break;

        case POWERPC_EXCP_DSTLB:    /* Data store TLB miss                   */

            cpu_abort(env, "Data store TLB exception while in user-mode. "

                      "Aborting");

            break;

        case POWERPC_EXCP_FPA:      /* Floating-point assist exception       */

            cpu_abort(env, "Floating-point assist exception not handled\n");

            break;

        case POWERPC_EXCP_IABR:     /* Instruction address breakpoint        */

            cpu_abort(env, "Instruction address breakpoint exception "

                      "not handled\n");

            break;

        case POWERPC_EXCP_SMI:      /* System management interrupt           */

            cpu_abort(env, "System management interrupt while in user mode. "

                      "Aborting\n");

            break;

        case POWERPC_EXCP_THERM:    /* Thermal interrupt                     */

            cpu_abort(env, "Thermal interrupt interrupt while in user mode. "

                      "Aborting\n");

            break;

        case POWERPC_EXCP_PERFM:   /* Embedded performance monitor IRQ      */

            cpu_abort(env, "Performance monitor exception not handled\n");

            break;

        case POWERPC_EXCP_VPUA:     /* Vector assist exception               */

            cpu_abort(env, "Vector assist exception not handled\n");

            break;

        case POWERPC_EXCP_SOFTP:    /* Soft patch exception                  */

            cpu_abort(env, "Soft patch exception not handled\n");

            break;

        case POWERPC_EXCP_MAINT:    /* Maintenance exception                 */

            cpu_abort(env, "Maintenance exception while in user mode. "

                      "Aborting\n");

            break;

        case POWERPC_EXCP_STOP:     /* stop translation                      */

            /* We did invalidate the instruction cache. Go on */

            break;

        case POWERPC_EXCP_BRANCH:   /* branch instruction:                   */

            /* We just stopped because of a branch. Go on */

            break;

        case POWERPC_EXCP_SYSCALL_USER:

            /* system call in user-mode emulation */

            /* WARNING:

             * PPC ABI uses overflow flag in cr0 to signal an error

             * in syscalls.

             */

#if 0

            printf("syscall %d 0x%08x 0x%08x 0x%08x 0x%08x\n", env->gpr[0],

                   env->gpr[3], env->gpr[4], env->gpr[5], env->gpr[6]);

#endif

            env->crf[0] &= ~0x1;

            ret = do_syscall(env, env->gpr[0], env->gpr[3], env->gpr[4],

                             env->gpr[5], env->gpr[6], env->gpr[7],

                             env->gpr[8]);

            if (ret > (uint32_t)(-515)) {

                env->crf[0] |= 0x1;

                ret = -ret;

            }

            env->gpr[3] = ret;

#if 0

            printf("syscall returned 0x%08x (%d)\n", ret, ret);

#endif

            break;

        case EXCP_INTERRUPT:

            /* just indicate that signals should be handled asap */

            break;

        default:

            cpu_abort(env, "Unknown exception 0x%d. Aborting\n", trapnr);

            break;

        }

        process_pending_signals(env);

    }

}
