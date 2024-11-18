int pic_read_irq(DeviceState *d)

{

    PICCommonState *s = DO_UPCAST(PICCommonState, dev.qdev, d);

    int irq, irq2, intno;



    irq = pic_get_irq(s);

    if (irq >= 0) {

        if (irq == 2) {

            irq2 = pic_get_irq(slave_pic);

            if (irq2 >= 0) {

                pic_intack(slave_pic, irq2);

            } else {

                /* spurious IRQ on slave controller */

                if (no_spurious_interrupt_hack) {

                    /* Pretend it was delivered and acknowledged.  If

                     * it was spurious due to slave_pic->imr, then

                     * as soon as the mask is cleared, the slave will

                     * re-trigger IRQ2 on the master.  If it is spurious for

                     * some other reason, make sure we don't keep trying

                     * to half-process the same spurious interrupt over

                     * and over again.

                     */

                    s->irr &= ~(1<<irq);

                    s->last_irr &= ~(1<<irq);

                    s->isr &= ~(1<<irq);

                    return -1;

                }

                irq2 = 7;

            }

            intno = slave_pic->irq_base + irq2;

        } else {

            intno = s->irq_base + irq;

        }

        pic_intack(s, irq);

    } else {

        /* spurious IRQ on host controller */

        if (no_spurious_interrupt_hack) {

            return -1;

        }

        irq = 7;

        intno = s->irq_base + irq;

    }



#if defined(DEBUG_PIC) || defined(DEBUG_IRQ_LATENCY)

    if (irq == 2) {

        irq = irq2 + 8;

    }

#endif

#ifdef DEBUG_IRQ_LATENCY

    printf("IRQ%d latency=%0.3fus\n",

           irq,

           (double)(qemu_get_clock_ns(vm_clock) -

                    irq_time[irq]) * 1000000.0 / get_ticks_per_sec());

#endif

    DPRINTF("pic_interrupt: irq=%d\n", irq);

    return intno;

}
