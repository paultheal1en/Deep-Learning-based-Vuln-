void ppc_set_irq (CPUState *env, int n_IRQ, int level)

{

    if (level) {

        env->pending_interrupts |= 1 << n_IRQ;

        cpu_interrupt(env, CPU_INTERRUPT_HARD);

    } else {

        env->pending_interrupts &= ~(1 << n_IRQ);

        if (env->pending_interrupts == 0)

            cpu_reset_interrupt(env, CPU_INTERRUPT_HARD);

    }

#if defined(PPC_DEBUG_IRQ)

    if (loglevel & CPU_LOG_INT) {

        fprintf(logfile, "%s: %p n_IRQ %d level %d => pending %08x req %08x\n",

                __func__, env, n_IRQ, level,

                env->pending_interrupts, env->interrupt_request);

    }

#endif

}
