void pxa27x_timer_init(target_phys_addr_t base,

                qemu_irq *irqs, qemu_irq irq4)

{

    pxa2xx_timer_info *s = pxa2xx_timer_init(base, irqs);

    int i;

    s->freq = PXA27X_FREQ;

    s->tm4 = (PXA2xxTimer4 *) qemu_mallocz(8 *

                    sizeof(PXA2xxTimer4));

    for (i = 0; i < 8; i ++) {

        s->tm4[i].tm.value = 0;

        s->tm4[i].tm.irq = irq4;

        s->tm4[i].tm.info = s;

        s->tm4[i].tm.num = i + 4;

        s->tm4[i].tm.level = 0;

        s->tm4[i].freq = 0;

        s->tm4[i].control = 0x0;

        s->tm4[i].tm.qtimer = qemu_new_timer(vm_clock,

                        pxa2xx_timer_tick4, &s->tm4[i]);

    }

}
