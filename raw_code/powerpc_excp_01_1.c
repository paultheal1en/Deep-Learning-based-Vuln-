static always_inline void powerpc_excp (CPUState *env,

                                        int excp_model, int excp)

{

    target_ulong msr, new_msr, vector;

    int srr0, srr1, asrr0, asrr1;

#if defined(TARGET_PPC64H)

    int lpes0, lpes1, lev;



    lpes0 = (env->spr[SPR_LPCR] >> 1) & 1;

    lpes1 = (env->spr[SPR_LPCR] >> 2) & 1;

#endif



    if (loglevel & CPU_LOG_INT) {

        fprintf(logfile, "Raise exception at 0x" ADDRX " => 0x%08x (%02x)\n",

                env->nip, excp, env->error_code);

    }

    msr = env->msr;

    new_msr = msr;

    srr0 = SPR_SRR0;

    srr1 = SPR_SRR1;

    asrr0 = -1;

    asrr1 = -1;

    msr &= ~((target_ulong)0x783F0000);

    switch (excp) {

    case POWERPC_EXCP_NONE:

        /* Should never happen */

        return;

    case POWERPC_EXCP_CRITICAL:    /* Critical input                         */

        new_msr &= ~((target_ulong)1 << MSR_RI); /* XXX: check this */

        switch (excp_model) {

        case POWERPC_EXCP_40x:

            srr0 = SPR_40x_SRR2;

            srr1 = SPR_40x_SRR3;

            break;

        case POWERPC_EXCP_BOOKE:

            srr0 = SPR_BOOKE_CSRR0;

            srr1 = SPR_BOOKE_CSRR1;

            break;

        case POWERPC_EXCP_G2:

            break;

        default:

            goto excp_invalid;

        }

        goto store_next;

    case POWERPC_EXCP_MCHECK:    /* Machine check exception                  */

        if (msr_me == 0) {

            /* Machine check exception is not enabled.

             * Enter checkstop state.

             */

            if (loglevel != 0) {

                fprintf(logfile, "Machine check while not allowed. "

                        "Entering checkstop state\n");

            } else {

                fprintf(stderr, "Machine check while not allowed. "

                        "Entering checkstop state\n");

            }

            env->halted = 1;

            env->interrupt_request |= CPU_INTERRUPT_EXITTB;

        }

        new_msr &= ~((target_ulong)1 << MSR_RI);

        new_msr &= ~((target_ulong)1 << MSR_ME);

#if defined(TARGET_PPC64H)

        new_msr |= (target_ulong)1 << MSR_HV;

#endif

        /* XXX: should also have something loaded in DAR / DSISR */

        switch (excp_model) {

        case POWERPC_EXCP_40x:

            srr0 = SPR_40x_SRR2;

            srr1 = SPR_40x_SRR3;

            break;

        case POWERPC_EXCP_BOOKE:

            srr0 = SPR_BOOKE_MCSRR0;

            srr1 = SPR_BOOKE_MCSRR1;

            asrr0 = SPR_BOOKE_CSRR0;

            asrr1 = SPR_BOOKE_CSRR1;

            break;

        default:

            break;

        }

        goto store_next;

    case POWERPC_EXCP_DSI:       /* Data storage exception                   */

#if defined (DEBUG_EXCEPTIONS)

        if (loglevel != 0) {

            fprintf(logfile, "DSI exception: DSISR=0x" ADDRX" DAR=0x" ADDRX

                    "\n", env->spr[SPR_DSISR], env->spr[SPR_DAR]);

        }

#endif

        new_msr &= ~((target_ulong)1 << MSR_RI);

#if defined(TARGET_PPC64H)

        if (lpes1 == 0)

            new_msr |= (target_ulong)1 << MSR_HV;

#endif

        goto store_next;

    case POWERPC_EXCP_ISI:       /* Instruction storage exception            */

#if defined (DEBUG_EXCEPTIONS)

        if (loglevel != 0) {

            fprintf(logfile, "ISI exception: msr=0x" ADDRX ", nip=0x" ADDRX

                    "\n", msr, env->nip);

        }

#endif

        new_msr &= ~((target_ulong)1 << MSR_RI);

#if defined(TARGET_PPC64H)

        if (lpes1 == 0)

            new_msr |= (target_ulong)1 << MSR_HV;

#endif

        msr |= env->error_code;

        goto store_next;

    case POWERPC_EXCP_EXTERNAL:  /* External input                           */

        new_msr &= ~((target_ulong)1 << MSR_RI);

#if defined(TARGET_PPC64H)

        if (lpes0 == 1)

            new_msr |= (target_ulong)1 << MSR_HV;

#endif

        goto store_next;

    case POWERPC_EXCP_ALIGN:     /* Alignment exception                      */

        new_msr &= ~((target_ulong)1 << MSR_RI);

#if defined(TARGET_PPC64H)

        if (lpes1 == 0)

            new_msr |= (target_ulong)1 << MSR_HV;

#endif

        /* XXX: this is false */

        /* Get rS/rD and rA from faulting opcode */

        env->spr[SPR_DSISR] |= (ldl_code((env->nip - 4)) & 0x03FF0000) >> 16;

        goto store_current;

    case POWERPC_EXCP_PROGRAM:   /* Program exception                        */

        switch (env->error_code & ~0xF) {

        case POWERPC_EXCP_FP:

            if ((msr_fe0 == 0 && msr_fe1 == 0) || msr_fp == 0) {

#if defined (DEBUG_EXCEPTIONS)

                if (loglevel != 0) {

                    fprintf(logfile, "Ignore floating point exception\n");

                }

#endif

                env->exception_index = POWERPC_EXCP_NONE;

                env->error_code = 0;

                return;

            }

            new_msr &= ~((target_ulong)1 << MSR_RI);

#if defined(TARGET_PPC64H)

            if (lpes1 == 0)

                new_msr |= (target_ulong)1 << MSR_HV;

#endif

            msr |= 0x00100000;

            if (msr_fe0 == msr_fe1)

                goto store_next;

            msr |= 0x00010000;

            break;

        case POWERPC_EXCP_INVAL:

#if defined (DEBUG_EXCEPTIONS)

            if (loglevel != 0) {

                fprintf(logfile, "Invalid instruction at 0x" ADDRX "\n",

                        env->nip);

            }

#endif

            new_msr &= ~((target_ulong)1 << MSR_RI);

#if defined(TARGET_PPC64H)

            if (lpes1 == 0)

                new_msr |= (target_ulong)1 << MSR_HV;

#endif

            msr |= 0x00080000;

            break;

        case POWERPC_EXCP_PRIV:

            new_msr &= ~((target_ulong)1 << MSR_RI);

#if defined(TARGET_PPC64H)

            if (lpes1 == 0)

                new_msr |= (target_ulong)1 << MSR_HV;

#endif

            msr |= 0x00040000;

            break;

        case POWERPC_EXCP_TRAP:

            new_msr &= ~((target_ulong)1 << MSR_RI);

#if defined(TARGET_PPC64H)

            if (lpes1 == 0)

                new_msr |= (target_ulong)1 << MSR_HV;

#endif

            msr |= 0x00020000;

            break;

        default:

            /* Should never occur */

            cpu_abort(env, "Invalid program exception %d. Aborting\n",

                      env->error_code);

            break;

        }

        goto store_current;

    case POWERPC_EXCP_FPU:       /* Floating-point unavailable exception     */

        new_msr &= ~((target_ulong)1 << MSR_RI);

#if defined(TARGET_PPC64H)

        if (lpes1 == 0)

            new_msr |= (target_ulong)1 << MSR_HV;

#endif

        goto store_current;

    case POWERPC_EXCP_SYSCALL:   /* System call exception                    */

        /* NOTE: this is a temporary hack to support graphics OSI

           calls from the MOL driver */

        /* XXX: To be removed */

        if (env->gpr[3] == 0x113724fa && env->gpr[4] == 0x77810f9b &&

            env->osi_call) {

            if (env->osi_call(env) != 0) {

                env->exception_index = POWERPC_EXCP_NONE;

                env->error_code = 0;

                return;

            }

        }

        if (loglevel & CPU_LOG_INT) {

            dump_syscall(env);

        }

        new_msr &= ~((target_ulong)1 << MSR_RI);

#if defined(TARGET_PPC64H)

        lev = env->error_code;

        if (lev == 1 || (lpes0 == 0 && lpes1 == 0))

            new_msr |= (target_ulong)1 << MSR_HV;

#endif

        goto store_next;

    case POWERPC_EXCP_APU:       /* Auxiliary processor unavailable          */

        new_msr &= ~((target_ulong)1 << MSR_RI);

        goto store_current;

    case POWERPC_EXCP_DECR:      /* Decrementer exception                    */

        new_msr &= ~((target_ulong)1 << MSR_RI);

#if defined(TARGET_PPC64H)

        if (lpes1 == 0)

            new_msr |= (target_ulong)1 << MSR_HV;

#endif

        goto store_next;

    case POWERPC_EXCP_FIT:       /* Fixed-interval timer interrupt           */

        /* FIT on 4xx */

#if defined (DEBUG_EXCEPTIONS)

        if (loglevel != 0)

            fprintf(logfile, "FIT exception\n");

#endif

        new_msr &= ~((target_ulong)1 << MSR_RI); /* XXX: check this */

        goto store_next;

    case POWERPC_EXCP_WDT:       /* Watchdog timer interrupt                 */

#if defined (DEBUG_EXCEPTIONS)

        if (loglevel != 0)

            fprintf(logfile, "WDT exception\n");

#endif

        switch (excp_model) {

        case POWERPC_EXCP_BOOKE:

            srr0 = SPR_BOOKE_CSRR0;

            srr1 = SPR_BOOKE_CSRR1;

            break;

        default:

            break;

        }

        new_msr &= ~((target_ulong)1 << MSR_RI); /* XXX: check this */

        goto store_next;

    case POWERPC_EXCP_DTLB:      /* Data TLB error                           */

        new_msr &= ~((target_ulong)1 << MSR_RI); /* XXX: check this */

        goto store_next;

    case POWERPC_EXCP_ITLB:      /* Instruction TLB error                    */

        new_msr &= ~((target_ulong)1 << MSR_RI); /* XXX: check this */

        goto store_next;

    case POWERPC_EXCP_DEBUG:     /* Debug interrupt                          */

        switch (excp_model) {

        case POWERPC_EXCP_BOOKE:

            srr0 = SPR_BOOKE_DSRR0;

            srr1 = SPR_BOOKE_DSRR1;

            asrr0 = SPR_BOOKE_CSRR0;

            asrr1 = SPR_BOOKE_CSRR1;

            break;

        default:

            break;

        }

        /* XXX: TODO */

        cpu_abort(env, "Debug exception is not implemented yet !\n");

        goto store_next;

#if defined(TARGET_PPCEMB)

    case POWERPC_EXCP_SPEU:      /* SPE/embedded floating-point unavailable  */

        new_msr &= ~((target_ulong)1 << MSR_RI); /* XXX: check this */

        goto store_current;

    case POWERPC_EXCP_EFPDI:     /* Embedded floating-point data interrupt   */

        /* XXX: TODO */

        cpu_abort(env, "Embedded floating point data exception "

                  "is not implemented yet !\n");

        goto store_next;

    case POWERPC_EXCP_EFPRI:     /* Embedded floating-point round interrupt  */

        /* XXX: TODO */

        cpu_abort(env, "Embedded floating point round exception "

                  "is not implemented yet !\n");

        goto store_next;

    case POWERPC_EXCP_EPERFM:    /* Embedded performance monitor interrupt   */

        new_msr &= ~((target_ulong)1 << MSR_RI);

        /* XXX: TODO */

        cpu_abort(env,

                  "Performance counter exception is not implemented yet !\n");

        goto store_next;

    case POWERPC_EXCP_DOORI:     /* Embedded doorbell interrupt              */

        /* XXX: TODO */

        cpu_abort(env,

                  "Embedded doorbell interrupt is not implemented yet !\n");

        goto store_next;

    case POWERPC_EXCP_DOORCI:    /* Embedded doorbell critical interrupt     */

        switch (excp_model) {

        case POWERPC_EXCP_BOOKE:

            srr0 = SPR_BOOKE_CSRR0;

            srr1 = SPR_BOOKE_CSRR1;

            break;

        default:

            break;

        }

        /* XXX: TODO */

        cpu_abort(env, "Embedded doorbell critical interrupt "

                  "is not implemented yet !\n");

        goto store_next;

#endif /* defined(TARGET_PPCEMB) */

    case POWERPC_EXCP_RESET:     /* System reset exception                   */

        new_msr &= ~((target_ulong)1 << MSR_RI);

#if defined(TARGET_PPC64H)

        new_msr |= (target_ulong)1 << MSR_HV;

#endif

        goto store_next;

#if defined(TARGET_PPC64)

    case POWERPC_EXCP_DSEG:      /* Data segment exception                   */

        new_msr &= ~((target_ulong)1 << MSR_RI);

#if defined(TARGET_PPC64H)

        if (lpes1 == 0)

            new_msr |= (target_ulong)1 << MSR_HV;

#endif

        goto store_next;

    case POWERPC_EXCP_ISEG:      /* Instruction segment exception            */

        new_msr &= ~((target_ulong)1 << MSR_RI);

#if defined(TARGET_PPC64H)

        if (lpes1 == 0)

            new_msr |= (target_ulong)1 << MSR_HV;

#endif

        goto store_next;

#endif /* defined(TARGET_PPC64) */

#if defined(TARGET_PPC64H)

    case POWERPC_EXCP_HDECR:     /* Hypervisor decrementer exception         */

        srr0 = SPR_HSRR0;

        srr1 = SPR_HSRR1;

        new_msr |= (target_ulong)1 << MSR_HV;

        goto store_next;

#endif

    case POWERPC_EXCP_TRACE:     /* Trace exception                          */

        new_msr &= ~((target_ulong)1 << MSR_RI);

#if defined(TARGET_PPC64H)

        if (lpes1 == 0)

            new_msr |= (target_ulong)1 << MSR_HV;

#endif

        goto store_next;

#if defined(TARGET_PPC64H)

    case POWERPC_EXCP_HDSI:      /* Hypervisor data storage exception        */

        srr0 = SPR_HSRR0;

        srr1 = SPR_HSRR1;

        new_msr |= (target_ulong)1 << MSR_HV;

        goto store_next;

    case POWERPC_EXCP_HISI:      /* Hypervisor instruction storage exception */

        srr0 = SPR_HSRR0;

        srr1 = SPR_HSRR1;

        new_msr |= (target_ulong)1 << MSR_HV;

        goto store_next;

    case POWERPC_EXCP_HDSEG:     /* Hypervisor data segment exception        */

        srr0 = SPR_HSRR0;

        srr1 = SPR_HSRR1;

        new_msr |= (target_ulong)1 << MSR_HV;

        goto store_next;

    case POWERPC_EXCP_HISEG:     /* Hypervisor instruction segment exception */

        srr0 = SPR_HSRR0;

        srr1 = SPR_HSRR1;

        new_msr |= (target_ulong)1 << MSR_HV;

        goto store_next;

#endif /* defined(TARGET_PPC64H) */

    case POWERPC_EXCP_VPU:       /* Vector unavailable exception             */

        new_msr &= ~((target_ulong)1 << MSR_RI);

#if defined(TARGET_PPC64H)

        if (lpes1 == 0)

            new_msr |= (target_ulong)1 << MSR_HV;

#endif

        goto store_current;

    case POWERPC_EXCP_PIT:       /* Programmable interval timer interrupt    */

#if defined (DEBUG_EXCEPTIONS)

        if (loglevel != 0)

            fprintf(logfile, "PIT exception\n");

#endif

        new_msr &= ~((target_ulong)1 << MSR_RI); /* XXX: check this */

        goto store_next;

    case POWERPC_EXCP_IO:        /* IO error exception                       */

        /* XXX: TODO */

        cpu_abort(env, "601 IO error exception is not implemented yet !\n");

        goto store_next;

    case POWERPC_EXCP_RUNM:      /* Run mode exception                       */

        /* XXX: TODO */

        cpu_abort(env, "601 run mode exception is not implemented yet !\n");

        goto store_next;

    case POWERPC_EXCP_EMUL:      /* Emulation trap exception                 */

        /* XXX: TODO */

        cpu_abort(env, "602 emulation trap exception "

                  "is not implemented yet !\n");

        goto store_next;

    case POWERPC_EXCP_IFTLB:     /* Instruction fetch TLB error              */

        new_msr &= ~((target_ulong)1 << MSR_RI); /* XXX: check this */

#if defined(TARGET_PPC64H) /* XXX: check this */

        if (lpes1 == 0)

            new_msr |= (target_ulong)1 << MSR_HV;

#endif

        switch (excp_model) {

        case POWERPC_EXCP_602:

        case POWERPC_EXCP_603:

        case POWERPC_EXCP_603E:

        case POWERPC_EXCP_G2:

            goto tlb_miss_tgpr;

        case POWERPC_EXCP_7x5:

            goto tlb_miss;

        case POWERPC_EXCP_74xx:

            goto tlb_miss_74xx;

        default:

            cpu_abort(env, "Invalid instruction TLB miss exception\n");

            break;

        }

        break;

    case POWERPC_EXCP_DLTLB:     /* Data load TLB miss                       */

        new_msr &= ~((target_ulong)1 << MSR_RI); /* XXX: check this */

#if defined(TARGET_PPC64H) /* XXX: check this */

        if (lpes1 == 0)

            new_msr |= (target_ulong)1 << MSR_HV;

#endif

        switch (excp_model) {

        case POWERPC_EXCP_602:

        case POWERPC_EXCP_603:

        case POWERPC_EXCP_603E:

        case POWERPC_EXCP_G2:

            goto tlb_miss_tgpr;

        case POWERPC_EXCP_7x5:

            goto tlb_miss;

        case POWERPC_EXCP_74xx:

            goto tlb_miss_74xx;

        default:

            cpu_abort(env, "Invalid data load TLB miss exception\n");

            break;

        }

        break;

    case POWERPC_EXCP_DSTLB:     /* Data store TLB miss                      */

        new_msr &= ~((target_ulong)1 << MSR_RI); /* XXX: check this */

#if defined(TARGET_PPC64H) /* XXX: check this */

        if (lpes1 == 0)

            new_msr |= (target_ulong)1 << MSR_HV;

#endif

        switch (excp_model) {

        case POWERPC_EXCP_602:

        case POWERPC_EXCP_603:

        case POWERPC_EXCP_603E:

        case POWERPC_EXCP_G2:

        tlb_miss_tgpr:

            /* Swap temporary saved registers with GPRs */

            if (!(new_msr & ((target_ulong)1 << MSR_TGPR))) {

                new_msr |= (target_ulong)1 << MSR_TGPR;

                hreg_swap_gpr_tgpr(env);

            }

            goto tlb_miss;

        case POWERPC_EXCP_7x5:

        tlb_miss:

#if defined (DEBUG_SOFTWARE_TLB)

            if (loglevel != 0) {

                const unsigned char *es;

                target_ulong *miss, *cmp;

                int en;

                if (excp == POWERPC_EXCP_IFTLB) {

                    es = "I";

                    en = 'I';

                    miss = &env->spr[SPR_IMISS];

                    cmp = &env->spr[SPR_ICMP];

                } else {

                    if (excp == POWERPC_EXCP_DLTLB)

                        es = "DL";

                    else

                        es = "DS";

                    en = 'D';

                    miss = &env->spr[SPR_DMISS];

                    cmp = &env->spr[SPR_DCMP];

                }

                fprintf(logfile, "6xx %sTLB miss: %cM " ADDRX " %cC " ADDRX

                        " H1 " ADDRX " H2 " ADDRX " %08x\n",

                        es, en, *miss, en, *cmp,

                        env->spr[SPR_HASH1], env->spr[SPR_HASH2],

                        env->error_code);

            }

#endif

            msr |= env->crf[0] << 28;

            msr |= env->error_code; /* key, D/I, S/L bits */

            /* Set way using a LRU mechanism */

            msr |= ((env->last_way + 1) & (env->nb_ways - 1)) << 17;

            break;

        case POWERPC_EXCP_74xx:

        tlb_miss_74xx:

#if defined (DEBUG_SOFTWARE_TLB)

            if (loglevel != 0) {

                const unsigned char *es;

                target_ulong *miss, *cmp;

                int en;

                if (excp == POWERPC_EXCP_IFTLB) {

                    es = "I";

                    en = 'I';

                    miss = &env->spr[SPR_TLBMISS];

                    cmp = &env->spr[SPR_PTEHI];

                } else {

                    if (excp == POWERPC_EXCP_DLTLB)

                        es = "DL";

                    else

                        es = "DS";

                    en = 'D';

                    miss = &env->spr[SPR_TLBMISS];

                    cmp = &env->spr[SPR_PTEHI];

                }

                fprintf(logfile, "74xx %sTLB miss: %cM " ADDRX " %cC " ADDRX

                        " %08x\n",

                        es, en, *miss, en, *cmp, env->error_code);

            }

#endif

            msr |= env->error_code; /* key bit */

            break;

        default:

            cpu_abort(env, "Invalid data store TLB miss exception\n");

            break;

        }

        goto store_next;

    case POWERPC_EXCP_FPA:       /* Floating-point assist exception          */

        /* XXX: TODO */

        cpu_abort(env, "Floating point assist exception "

                  "is not implemented yet !\n");

        goto store_next;

    case POWERPC_EXCP_IABR:      /* Instruction address breakpoint           */

        /* XXX: TODO */

        cpu_abort(env, "IABR exception is not implemented yet !\n");

        goto store_next;

    case POWERPC_EXCP_SMI:       /* System management interrupt              */

        /* XXX: TODO */

        cpu_abort(env, "SMI exception is not implemented yet !\n");

        goto store_next;

    case POWERPC_EXCP_THERM:     /* Thermal interrupt                        */

        /* XXX: TODO */

        cpu_abort(env, "Thermal management exception "

                  "is not implemented yet !\n");

        goto store_next;

    case POWERPC_EXCP_PERFM:     /* Embedded performance monitor interrupt   */

        new_msr &= ~((target_ulong)1 << MSR_RI);

#if defined(TARGET_PPC64H)

        if (lpes1 == 0)

            new_msr |= (target_ulong)1 << MSR_HV;

#endif

        /* XXX: TODO */

        cpu_abort(env,

                  "Performance counter exception is not implemented yet !\n");

        goto store_next;

    case POWERPC_EXCP_VPUA:      /* Vector assist exception                  */

        /* XXX: TODO */

        cpu_abort(env, "VPU assist exception is not implemented yet !\n");

        goto store_next;

    case POWERPC_EXCP_SOFTP:     /* Soft patch exception                     */

        /* XXX: TODO */

        cpu_abort(env,

                  "970 soft-patch exception is not implemented yet !\n");

        goto store_next;

    case POWERPC_EXCP_MAINT:     /* Maintenance exception                    */

        /* XXX: TODO */

        cpu_abort(env,

                  "970 maintenance exception is not implemented yet !\n");

        goto store_next;

    default:

    excp_invalid:

        cpu_abort(env, "Invalid PowerPC exception %d. Aborting\n", excp);

        break;

    store_current:

        /* save current instruction location */

        env->spr[srr0] = env->nip - 4;

        break;

    store_next:

        /* save next instruction location */

        env->spr[srr0] = env->nip;

        break;

    }

    /* Save MSR */

    env->spr[srr1] = msr;

    /* If any alternate SRR register are defined, duplicate saved values */

    if (asrr0 != -1)

        env->spr[asrr0] = env->spr[srr0];

    if (asrr1 != -1)

        env->spr[asrr1] = env->spr[srr1];

    /* If we disactivated any translation, flush TLBs */

    if (new_msr & ((1 << MSR_IR) | (1 << MSR_DR)))

        tlb_flush(env, 1);

    /* reload MSR with correct bits */

    new_msr &= ~((target_ulong)1 << MSR_EE);

    new_msr &= ~((target_ulong)1 << MSR_PR);

    new_msr &= ~((target_ulong)1 << MSR_FP);

    new_msr &= ~((target_ulong)1 << MSR_FE0);

    new_msr &= ~((target_ulong)1 << MSR_SE);

    new_msr &= ~((target_ulong)1 << MSR_BE);

    new_msr &= ~((target_ulong)1 << MSR_FE1);

    new_msr &= ~((target_ulong)1 << MSR_IR);

    new_msr &= ~((target_ulong)1 << MSR_DR);

#if 0 /* Fix this: not on all targets */

    new_msr &= ~((target_ulong)1 << MSR_PMM);

#endif

    new_msr &= ~((target_ulong)1 << MSR_LE);

    if (msr_ile)

        new_msr |= (target_ulong)1 << MSR_LE;

    else

        new_msr &= ~((target_ulong)1 << MSR_LE);

    /* Jump to handler */

    vector = env->excp_vectors[excp];

    if (vector == (target_ulong)-1) {

        cpu_abort(env, "Raised an exception without defined vector %d\n",

                  excp);

    }

    vector |= env->excp_prefix;

#if defined(TARGET_PPC64)

    if (excp_model == POWERPC_EXCP_BOOKE) {

        if (!msr_icm) {

            new_msr &= ~((target_ulong)1 << MSR_CM);

            vector = (uint32_t)vector;

        } else {

            new_msr |= (target_ulong)1 << MSR_CM;

        }

    } else {

        if (!msr_isf) {

            new_msr &= ~((target_ulong)1 << MSR_SF);

            vector = (uint32_t)vector;

        } else {

            new_msr |= (target_ulong)1 << MSR_SF;

        }

    }

#endif

    /* XXX: we don't use hreg_store_msr here as already have treated

     *      any special case that could occur. Just store MSR and update hflags

     */

    env->msr = new_msr;

    env->hflags_nmsr = 0x00000000;

    hreg_compute_hflags(env);

    env->nip = vector;

    /* Reset exception state */

    env->exception_index = POWERPC_EXCP_NONE;

    env->error_code = 0;

}
