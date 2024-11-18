int cpu_exec(CPUArchState *env)

{

    CPUState *cpu = ENV_GET_CPU(env);

#if !(defined(CONFIG_USER_ONLY) && \

      (defined(TARGET_M68K) || defined(TARGET_PPC) || defined(TARGET_S390X)))

    CPUClass *cc = CPU_GET_CLASS(cpu);

#endif

#ifdef TARGET_I386

    X86CPU *x86_cpu = X86_CPU(cpu);

#endif

    int ret, interrupt_request;

    TranslationBlock *tb;

    uint8_t *tc_ptr;

    uintptr_t next_tb;





    if (cpu->halted) {

        if (!cpu_has_work(cpu)) {

            return EXCP_HALTED;

        }



        cpu->halted = 0;

    }



    current_cpu = cpu;



    /* As long as current_cpu is null, up to the assignment just above,

     * requests by other threads to exit the execution loop are expected to

     * be issued using the exit_request global. We must make sure that our

     * evaluation of the global value is performed past the current_cpu

     * value transition point, which requires a memory barrier as well as

     * an instruction scheduling constraint on modern architectures.  */

    smp_mb();



    if (unlikely(exit_request)) {

        cpu->exit_request = 1;

    }



#if defined(TARGET_I386)

    /* put eflags in CPU temporary format */

    CC_SRC = env->eflags & (CC_O | CC_S | CC_Z | CC_A | CC_P | CC_C);

    env->df = 1 - (2 * ((env->eflags >> 10) & 1));

    CC_OP = CC_OP_EFLAGS;

    env->eflags &= ~(DF_MASK | CC_O | CC_S | CC_Z | CC_A | CC_P | CC_C);

#elif defined(TARGET_SPARC)

#elif defined(TARGET_M68K)

    env->cc_op = CC_OP_FLAGS;

    env->cc_dest = env->sr & 0xf;

    env->cc_x = (env->sr >> 4) & 1;

#elif defined(TARGET_ALPHA)

#elif defined(TARGET_ARM)

#elif defined(TARGET_UNICORE32)

#elif defined(TARGET_PPC)

    env->reserve_addr = -1;

#elif defined(TARGET_LM32)

#elif defined(TARGET_MICROBLAZE)

#elif defined(TARGET_MIPS)

#elif defined(TARGET_MOXIE)

#elif defined(TARGET_OPENRISC)

#elif defined(TARGET_SH4)

#elif defined(TARGET_CRIS)

#elif defined(TARGET_S390X)

#elif defined(TARGET_XTENSA)

    /* XXXXX */

#else

#error unsupported target CPU

#endif

    cpu->exception_index = -1;



    /* prepare setjmp context for exception handling */

    for(;;) {

        if (sigsetjmp(cpu->jmp_env, 0) == 0) {

            /* if an exception is pending, we execute it here */

            if (cpu->exception_index >= 0) {

                if (cpu->exception_index >= EXCP_INTERRUPT) {

                    /* exit request from the cpu execution loop */

                    ret = cpu->exception_index;

                    if (ret == EXCP_DEBUG) {

                        cpu_handle_debug_exception(env);

                    }

                    break;

                } else {

#if defined(CONFIG_USER_ONLY)

                    /* if user mode only, we simulate a fake exception

                       which will be handled outside the cpu execution

                       loop */

#if defined(TARGET_I386)

                    cc->do_interrupt(cpu);

#endif

                    ret = cpu->exception_index;

                    break;

#else

                    cc->do_interrupt(cpu);

                    cpu->exception_index = -1;

#endif

                }

            }



            next_tb = 0; /* force lookup of first TB */

            for(;;) {

                interrupt_request = cpu->interrupt_request;

                if (unlikely(interrupt_request)) {

                    if (unlikely(cpu->singlestep_enabled & SSTEP_NOIRQ)) {

                        /* Mask out external interrupts for this step. */

                        interrupt_request &= ~CPU_INTERRUPT_SSTEP_MASK;

                    }

                    if (interrupt_request & CPU_INTERRUPT_DEBUG) {

                        cpu->interrupt_request &= ~CPU_INTERRUPT_DEBUG;

                        cpu->exception_index = EXCP_DEBUG;

                        cpu_loop_exit(cpu);

                    }

#if defined(TARGET_ARM) || defined(TARGET_SPARC) || defined(TARGET_MIPS) || \

    defined(TARGET_PPC) || defined(TARGET_ALPHA) || defined(TARGET_CRIS) || \

    defined(TARGET_MICROBLAZE) || defined(TARGET_LM32) || defined(TARGET_UNICORE32)

                    if (interrupt_request & CPU_INTERRUPT_HALT) {

                        cpu->interrupt_request &= ~CPU_INTERRUPT_HALT;

                        cpu->halted = 1;

                        cpu->exception_index = EXCP_HLT;

                        cpu_loop_exit(cpu);

                    }

#endif

#if defined(TARGET_I386)

#if !defined(CONFIG_USER_ONLY)

                    if (interrupt_request & CPU_INTERRUPT_POLL) {

                        cpu->interrupt_request &= ~CPU_INTERRUPT_POLL;

                        apic_poll_irq(x86_cpu->apic_state);

                    }

#endif

                    if (interrupt_request & CPU_INTERRUPT_INIT) {

                            cpu_svm_check_intercept_param(env, SVM_EXIT_INIT,

                                                          0);

                            do_cpu_init(x86_cpu);

                            cpu->exception_index = EXCP_HALTED;

                            cpu_loop_exit(cpu);

                    } else if (interrupt_request & CPU_INTERRUPT_SIPI) {

                            do_cpu_sipi(x86_cpu);

                    } else if (env->hflags2 & HF2_GIF_MASK) {

                        if ((interrupt_request & CPU_INTERRUPT_SMI) &&

                            !(env->hflags & HF_SMM_MASK)) {

                            cpu_svm_check_intercept_param(env, SVM_EXIT_SMI,

                                                          0);

                            cpu->interrupt_request &= ~CPU_INTERRUPT_SMI;

                            do_smm_enter(x86_cpu);

                            next_tb = 0;

                        } else if ((interrupt_request & CPU_INTERRUPT_NMI) &&

                                   !(env->hflags2 & HF2_NMI_MASK)) {

                            cpu->interrupt_request &= ~CPU_INTERRUPT_NMI;

                            env->hflags2 |= HF2_NMI_MASK;

                            do_interrupt_x86_hardirq(env, EXCP02_NMI, 1);

                            next_tb = 0;

                        } else if (interrupt_request & CPU_INTERRUPT_MCE) {

                            cpu->interrupt_request &= ~CPU_INTERRUPT_MCE;

                            do_interrupt_x86_hardirq(env, EXCP12_MCHK, 0);

                            next_tb = 0;

                        } else if ((interrupt_request & CPU_INTERRUPT_HARD) &&

                                   (((env->hflags2 & HF2_VINTR_MASK) && 

                                     (env->hflags2 & HF2_HIF_MASK)) ||

                                    (!(env->hflags2 & HF2_VINTR_MASK) && 

                                     (env->eflags & IF_MASK && 

                                      !(env->hflags & HF_INHIBIT_IRQ_MASK))))) {

                            int intno;

                            cpu_svm_check_intercept_param(env, SVM_EXIT_INTR,

                                                          0);

                            cpu->interrupt_request &= ~(CPU_INTERRUPT_HARD |

                                                        CPU_INTERRUPT_VIRQ);

                            intno = cpu_get_pic_interrupt(env);

                            qemu_log_mask(CPU_LOG_TB_IN_ASM, "Servicing hardware INT=0x%02x\n", intno);

                            do_interrupt_x86_hardirq(env, intno, 1);

                            /* ensure that no TB jump will be modified as

                               the program flow was changed */

                            next_tb = 0;

#if !defined(CONFIG_USER_ONLY)

                        } else if ((interrupt_request & CPU_INTERRUPT_VIRQ) &&

                                   (env->eflags & IF_MASK) && 

                                   !(env->hflags & HF_INHIBIT_IRQ_MASK)) {

                            int intno;

                            /* FIXME: this should respect TPR */

                            cpu_svm_check_intercept_param(env, SVM_EXIT_VINTR,

                                                          0);

                            intno = ldl_phys(cpu->as,

                                             env->vm_vmcb

                                             + offsetof(struct vmcb,

                                                        control.int_vector));

                            qemu_log_mask(CPU_LOG_TB_IN_ASM, "Servicing virtual hardware INT=0x%02x\n", intno);

                            do_interrupt_x86_hardirq(env, intno, 1);

                            cpu->interrupt_request &= ~CPU_INTERRUPT_VIRQ;

                            next_tb = 0;

#endif

                        }

                    }

#elif defined(TARGET_PPC)

                    if ((interrupt_request & CPU_INTERRUPT_RESET)) {

                        cpu_reset(cpu);

                    }

                    if (interrupt_request & CPU_INTERRUPT_HARD) {

                        ppc_hw_interrupt(env);

                        if (env->pending_interrupts == 0) {

                            cpu->interrupt_request &= ~CPU_INTERRUPT_HARD;

                        }

                        next_tb = 0;

                    }

#elif defined(TARGET_LM32)

                    if ((interrupt_request & CPU_INTERRUPT_HARD)

                        && (env->ie & IE_IE)) {

                        cpu->exception_index = EXCP_IRQ;

                        cc->do_interrupt(cpu);

                        next_tb = 0;

                    }

#elif defined(TARGET_MICROBLAZE)

                    if ((interrupt_request & CPU_INTERRUPT_HARD)

                        && (env->sregs[SR_MSR] & MSR_IE)

                        && !(env->sregs[SR_MSR] & (MSR_EIP | MSR_BIP))

                        && !(env->iflags & (D_FLAG | IMM_FLAG))) {

                        cpu->exception_index = EXCP_IRQ;

                        cc->do_interrupt(cpu);

                        next_tb = 0;

                    }

#elif defined(TARGET_MIPS)

                    if ((interrupt_request & CPU_INTERRUPT_HARD) &&

                        cpu_mips_hw_interrupts_pending(env)) {

                        /* Raise it */

                        cpu->exception_index = EXCP_EXT_INTERRUPT;

                        env->error_code = 0;

                        cc->do_interrupt(cpu);

                        next_tb = 0;

                    }

#elif defined(TARGET_OPENRISC)

                    {

                        int idx = -1;

                        if ((interrupt_request & CPU_INTERRUPT_HARD)

                            && (env->sr & SR_IEE)) {

                            idx = EXCP_INT;

                        }

                        if ((interrupt_request & CPU_INTERRUPT_TIMER)

                            && (env->sr & SR_TEE)) {

                            idx = EXCP_TICK;

                        }

                        if (idx >= 0) {

                            cpu->exception_index = idx;

                            cc->do_interrupt(cpu);

                            next_tb = 0;

                        }

                    }

#elif defined(TARGET_SPARC)

                    if (interrupt_request & CPU_INTERRUPT_HARD) {

                        if (cpu_interrupts_enabled(env) &&

                            env->interrupt_index > 0) {

                            int pil = env->interrupt_index & 0xf;

                            int type = env->interrupt_index & 0xf0;



                            if (((type == TT_EXTINT) &&

                                  cpu_pil_allowed(env, pil)) ||

                                  type != TT_EXTINT) {

                                cpu->exception_index = env->interrupt_index;

                                cc->do_interrupt(cpu);

                                next_tb = 0;

                            }

                        }

                    }

#elif defined(TARGET_ARM)

                    if (interrupt_request & CPU_INTERRUPT_FIQ

                        && !(env->daif & PSTATE_F)) {

                        cpu->exception_index = EXCP_FIQ;

                        cc->do_interrupt(cpu);

                        next_tb = 0;

                    }

                    /* ARMv7-M interrupt return works by loading a magic value

                       into the PC.  On real hardware the load causes the

                       return to occur.  The qemu implementation performs the

                       jump normally, then does the exception return when the

                       CPU tries to execute code at the magic address.

                       This will cause the magic PC value to be pushed to

                       the stack if an interrupt occurred at the wrong time.

                       We avoid this by disabling interrupts when

                       pc contains a magic address.  */

                    if (interrupt_request & CPU_INTERRUPT_HARD

                        && ((IS_M(env) && env->regs[15] < 0xfffffff0)

                            || !(env->daif & PSTATE_I))) {

                        cpu->exception_index = EXCP_IRQ;

                        cc->do_interrupt(cpu);

                        next_tb = 0;

                    }

#elif defined(TARGET_UNICORE32)

                    if (interrupt_request & CPU_INTERRUPT_HARD

                        && !(env->uncached_asr & ASR_I)) {

                        cpu->exception_index = UC32_EXCP_INTR;

                        cc->do_interrupt(cpu);

                        next_tb = 0;

                    }

#elif defined(TARGET_SH4)

                    if (interrupt_request & CPU_INTERRUPT_HARD) {

                        cc->do_interrupt(cpu);

                        next_tb = 0;

                    }

#elif defined(TARGET_ALPHA)

                    {

                        int idx = -1;

                        /* ??? This hard-codes the OSF/1 interrupt levels.  */

                        switch (env->pal_mode ? 7 : env->ps & PS_INT_MASK) {

                        case 0 ... 3:

                            if (interrupt_request & CPU_INTERRUPT_HARD) {

                                idx = EXCP_DEV_INTERRUPT;

                            }

                            /* FALLTHRU */

                        case 4:

                            if (interrupt_request & CPU_INTERRUPT_TIMER) {

                                idx = EXCP_CLK_INTERRUPT;

                            }

                            /* FALLTHRU */

                        case 5:

                            if (interrupt_request & CPU_INTERRUPT_SMP) {

                                idx = EXCP_SMP_INTERRUPT;

                            }

                            /* FALLTHRU */

                        case 6:

                            if (interrupt_request & CPU_INTERRUPT_MCHK) {

                                idx = EXCP_MCHK;

                            }

                        }

                        if (idx >= 0) {

                            cpu->exception_index = idx;

                            env->error_code = 0;

                            cc->do_interrupt(cpu);

                            next_tb = 0;

                        }

                    }

#elif defined(TARGET_CRIS)

                    if (interrupt_request & CPU_INTERRUPT_HARD

                        && (env->pregs[PR_CCS] & I_FLAG)

                        && !env->locked_irq) {

                        cpu->exception_index = EXCP_IRQ;

                        cc->do_interrupt(cpu);

                        next_tb = 0;

                    }

                    if (interrupt_request & CPU_INTERRUPT_NMI) {

                        unsigned int m_flag_archval;

                        if (env->pregs[PR_VR] < 32) {

                            m_flag_archval = M_FLAG_V10;

                        } else {

                            m_flag_archval = M_FLAG_V32;

                        }

                        if ((env->pregs[PR_CCS] & m_flag_archval)) {

                            cpu->exception_index = EXCP_NMI;

                            cc->do_interrupt(cpu);

                            next_tb = 0;

                        }

                    }

#elif defined(TARGET_M68K)

                    if (interrupt_request & CPU_INTERRUPT_HARD

                        && ((env->sr & SR_I) >> SR_I_SHIFT)

                            < env->pending_level) {

                        /* Real hardware gets the interrupt vector via an

                           IACK cycle at this point.  Current emulated

                           hardware doesn't rely on this, so we

                           provide/save the vector when the interrupt is

                           first signalled.  */

                        cpu->exception_index = env->pending_vector;

                        do_interrupt_m68k_hardirq(env);

                        next_tb = 0;

                    }

#elif defined(TARGET_S390X) && !defined(CONFIG_USER_ONLY)

                    if ((interrupt_request & CPU_INTERRUPT_HARD) &&

                        (env->psw.mask & PSW_MASK_EXT)) {

                        cc->do_interrupt(cpu);

                        next_tb = 0;

                    }

#elif defined(TARGET_XTENSA)

                    if (interrupt_request & CPU_INTERRUPT_HARD) {

                        cpu->exception_index = EXC_IRQ;

                        cc->do_interrupt(cpu);

                        next_tb = 0;

                    }

#endif

                   /* Don't use the cached interrupt_request value,

                      do_interrupt may have updated the EXITTB flag. */

                    if (cpu->interrupt_request & CPU_INTERRUPT_EXITTB) {

                        cpu->interrupt_request &= ~CPU_INTERRUPT_EXITTB;

                        /* ensure that no TB jump will be modified as

                           the program flow was changed */

                        next_tb = 0;

                    }

                }

                if (unlikely(cpu->exit_request)) {

                    cpu->exit_request = 0;

                    cpu->exception_index = EXCP_INTERRUPT;

                    cpu_loop_exit(cpu);

                }

                spin_lock(&tcg_ctx.tb_ctx.tb_lock);

                have_tb_lock = true;

                tb = tb_find_fast(env);

                /* Note: we do it here to avoid a gcc bug on Mac OS X when

                   doing it in tb_find_slow */

                if (tcg_ctx.tb_ctx.tb_invalidated_flag) {

                    /* as some TB could have been invalidated because

                       of memory exceptions while generating the code, we

                       must recompute the hash index here */

                    next_tb = 0;

                    tcg_ctx.tb_ctx.tb_invalidated_flag = 0;

                }

                if (qemu_loglevel_mask(CPU_LOG_EXEC)) {

                    qemu_log("Trace %p [" TARGET_FMT_lx "] %s\n",

                             tb->tc_ptr, tb->pc, lookup_symbol(tb->pc));

                }

                /* see if we can patch the calling TB. When the TB

                   spans two pages, we cannot safely do a direct

                   jump. */

                if (next_tb != 0 && tb->page_addr[1] == -1) {

                    tb_add_jump((TranslationBlock *)(next_tb & ~TB_EXIT_MASK),

                                next_tb & TB_EXIT_MASK, tb);

                }

                have_tb_lock = false;

                spin_unlock(&tcg_ctx.tb_ctx.tb_lock);



                /* cpu_interrupt might be called while translating the

                   TB, but before it is linked into a potentially

                   infinite loop and becomes env->current_tb. Avoid

                   starting execution if there is a pending interrupt. */

                cpu->current_tb = tb;

                barrier();

                if (likely(!cpu->exit_request)) {

                    tc_ptr = tb->tc_ptr;

                    /* execute the generated code */

                    next_tb = cpu_tb_exec(cpu, tc_ptr);

                    switch (next_tb & TB_EXIT_MASK) {

                    case TB_EXIT_REQUESTED:

                        /* Something asked us to stop executing

                         * chained TBs; just continue round the main

                         * loop. Whatever requested the exit will also

                         * have set something else (eg exit_request or

                         * interrupt_request) which we will handle

                         * next time around the loop.

                         */

                        tb = (TranslationBlock *)(next_tb & ~TB_EXIT_MASK);

                        next_tb = 0;

                        break;

                    case TB_EXIT_ICOUNT_EXPIRED:

                    {

                        /* Instruction counter expired.  */

                        int insns_left;

                        tb = (TranslationBlock *)(next_tb & ~TB_EXIT_MASK);

                        insns_left = cpu->icount_decr.u32;

                        if (cpu->icount_extra && insns_left >= 0) {

                            /* Refill decrementer and continue execution.  */

                            cpu->icount_extra += insns_left;

                            if (cpu->icount_extra > 0xffff) {

                                insns_left = 0xffff;

                            } else {

                                insns_left = cpu->icount_extra;

                            }

                            cpu->icount_extra -= insns_left;

                            cpu->icount_decr.u16.low = insns_left;

                        } else {

                            if (insns_left > 0) {

                                /* Execute remaining instructions.  */

                                cpu_exec_nocache(env, insns_left, tb);

                            }

                            cpu->exception_index = EXCP_INTERRUPT;

                            next_tb = 0;

                            cpu_loop_exit(cpu);

                        }

                        break;

                    }

                    default:

                        break;

                    }

                }

                cpu->current_tb = NULL;

                /* reset soft MMU for next block (it can currently

                   only be set by a memory fault) */

            } /* for(;;) */

        } else {

            /* Reload env after longjmp - the compiler may have smashed all

             * local variables as longjmp is marked 'noreturn'. */

            cpu = current_cpu;

            env = cpu->env_ptr;

#if !(defined(CONFIG_USER_ONLY) && \

      (defined(TARGET_M68K) || defined(TARGET_PPC) || defined(TARGET_S390X)))

            cc = CPU_GET_CLASS(cpu);

#endif

#ifdef TARGET_I386

            x86_cpu = X86_CPU(cpu);

#endif

            if (have_tb_lock) {

                spin_unlock(&tcg_ctx.tb_ctx.tb_lock);

                have_tb_lock = false;

            }

        }

    } /* for(;;) */





#if defined(TARGET_I386)

    /* restore flags in standard format */

    env->eflags = env->eflags | cpu_cc_compute_all(env, CC_OP)

        | (env->df & DF_MASK);

#elif defined(TARGET_ARM)

    /* XXX: Save/restore host fpu exception state?.  */

#elif defined(TARGET_UNICORE32)

#elif defined(TARGET_SPARC)

#elif defined(TARGET_PPC)

#elif defined(TARGET_LM32)

#elif defined(TARGET_M68K)

    cpu_m68k_flush_flags(env, env->cc_op);

    env->cc_op = CC_OP_FLAGS;

    env->sr = (env->sr & 0xffe0)

              | env->cc_dest | (env->cc_x << 4);

#elif defined(TARGET_MICROBLAZE)

#elif defined(TARGET_MIPS)

#elif defined(TARGET_MOXIE)

#elif defined(TARGET_OPENRISC)

#elif defined(TARGET_SH4)

#elif defined(TARGET_ALPHA)

#elif defined(TARGET_CRIS)

#elif defined(TARGET_S390X)

#elif defined(TARGET_XTENSA)

    /* XXXXX */

#else

#error unsupported target CPU

#endif



    /* fail safe : never use current_cpu outside cpu_exec() */

    current_cpu = NULL;

    return ret;

}