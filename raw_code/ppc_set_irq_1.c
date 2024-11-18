void ppc_set_irq(PowerPCCPU *cpu, int n_IRQ, int level)

{

    CPUState *cs = CPU(cpu);

    CPUPPCState *env = &cpu->env;

    unsigned int old_pending = env->pending_interrupts;



    if (level) {

        env->pending_interrupts |= 1 << n_IRQ;

        cpu_interrupt(cs, CPU_INTERRUPT_HARD);

    } else {

        env->pending_interrupts &= ~(1 << n_IRQ);

        if (env->pending_interrupts == 0) {

            cpu_reset_interrupt(cs, CPU_INTERRUPT_HARD);

        }

    }



    if (old_pending != env->pending_interrupts) {

#ifdef CONFIG_KVM

        kvmppc_set_interrupt(cpu, n_IRQ, level);

#endif

    }



    LOG_IRQ("%s: %p n_IRQ %d level %d => pending %08" PRIx32

                "req %08x\n", __func__, env, n_IRQ, level,

                env->pending_interrupts, CPU(cpu)->interrupt_request);

}
