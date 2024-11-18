static void pxa2xx_gpio_set(void *opaque, int line, int level)

{

    PXA2xxGPIOInfo *s = (PXA2xxGPIOInfo *) opaque;

    CPUState *cpu = CPU(s->cpu);

    int bank;

    uint32_t mask;



    if (line >= s->lines) {

        printf("%s: No GPIO pin %i\n", __FUNCTION__, line);

        return;

    }



    bank = line >> 5;

    mask = 1U << (line & 31);



    if (level) {

        s->status[bank] |= s->rising[bank] & mask &

                ~s->ilevel[bank] & ~s->dir[bank];

        s->ilevel[bank] |= mask;

    } else {

        s->status[bank] |= s->falling[bank] & mask &

                s->ilevel[bank] & ~s->dir[bank];

        s->ilevel[bank] &= ~mask;

    }



    if (s->status[bank] & mask)

        pxa2xx_gpio_irq_update(s);



    /* Wake-up GPIOs */

    if (cpu->halted && (mask & ~s->dir[bank] & pxa2xx_gpio_wake[bank])) {

        cpu_interrupt(cpu, CPU_INTERRUPT_EXITTB);

    }

}
