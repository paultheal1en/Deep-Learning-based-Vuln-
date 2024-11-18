static void openrisc_pic_cpu_handler(void *opaque, int irq, int level)

{

    OpenRISCCPU *cpu = (OpenRISCCPU *)opaque;

    CPUState *cs = CPU(cpu);

    uint32_t irq_bit = 1 << irq;



    if (irq > 31 || irq < 0) {

        return;

    }



    if (level) {

        cpu->env.picsr |= irq_bit;

    } else {

        cpu->env.picsr &= ~irq_bit;

    }



    if (cpu->env.picsr & cpu->env.picmr) {

        cpu_interrupt(cs, CPU_INTERRUPT_HARD);

    } else {

        cpu_reset_interrupt(cs, CPU_INTERRUPT_HARD);

        cpu->env.picsr = 0;

    }

}
