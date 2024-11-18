int cpu_exec(CPUArchState *env)

{

    CPUState *cpu = ENV_GET_CPU(env);

    CPUClass *cc = CPU_GET_CLASS(cpu);

#ifdef TARGET_I386

    X86CPU *x86_cpu = X86_CPU(cpu);

#endif

    int ret, interrupt_request;

    TranslationBlock *tb;

    uint8_t *tc_ptr;

    uintptr_t next_tb;

    SyncClocks sc;



    /* This must be volatile so it is not trashed by longjmp() */

    volatile bool have_tb_lock = false;



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



    cc->cpu_exec_enter(cpu);

    cpu->exception_index = -1;



    /* Calculate difference between guest clock and host clock.

     * This delay includes the delay of the last cycle, so

     * what we have to do is sleep until it is 0. As for the

     * advance/delay we gain here, we try to fix it next time.

     */

    init_delay_params(&sc, cpu);



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

                    if (interrupt_request & CPU_INTERRUPT_HALT) {

                        cpu->interrupt_request &= ~CPU_INTERRUPT_HALT;

                        cpu->halted = 1;

                        cpu->exception_index = EXCP_HLT;

                        cpu_loop_exit(cpu);

                    }

#if defined(TARGET_I386)

                    if (interrupt_request & CPU_INTERRUPT_INIT) {

                        cpu_svm_check_intercept_param(env, SVM_EXIT_INIT, 0);

                        do_cpu_init(x86_cpu);

                        cpu->exception_index = EXCP_HALTED;

                        cpu_loop_exit(cpu);

                    }

#else

                    if (interrupt_request & CPU_INTERRUPT_RESET) {

                        cpu_reset(cpu);

                    }

#endif

                    /* The target hook has 3 exit conditions:

                       False when the interrupt isn't processed,

                       True when it is, and we should restart on a new TB,

                       and via longjmp via cpu_loop_exit.  */

                    if (cc->cpu_exec_interrupt(cpu, interrupt_request)) {

                        next_tb = 0;

                    }

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

                    trace_exec_tb(tb, tb->pc);

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

                                align_clocks(&sc, cpu);

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

                /* Try to align the host and virtual clocks

                   if the guest is in advance */

                align_clocks(&sc, cpu);

                /* reset soft MMU for next block (it can currently

                   only be set by a memory fault) */

            } /* for(;;) */

        } else {

            /* Reload env after longjmp - the compiler may have smashed all

             * local variables as longjmp is marked 'noreturn'. */

            cpu = current_cpu;

            env = cpu->env_ptr;

            cc = CPU_GET_CLASS(cpu);

#ifdef TARGET_I386

            x86_cpu = X86_CPU(cpu);

#endif

            if (have_tb_lock) {

                spin_unlock(&tcg_ctx.tb_ctx.tb_lock);

                have_tb_lock = false;

            }

        }

    } /* for(;;) */



    cc->cpu_exec_exit(cpu);



    /* fail safe : never use current_cpu outside cpu_exec() */

    current_cpu = NULL;

    return ret;

}
