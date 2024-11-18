static void cpu_set_irq(void *opaque, int irq, int level)

{

    CPUState *env = opaque;



    if (level) {

        CPUIRQ_DPRINTF("Raise CPU IRQ %d\n", irq);

        env->halted = 0;

        env->pil_in |= 1 << irq;

        cpu_check_irqs(env);

    } else {

        CPUIRQ_DPRINTF("Lower CPU IRQ %d\n", irq);

        env->pil_in &= ~(1 << irq);

        cpu_check_irqs(env);

    }

}
