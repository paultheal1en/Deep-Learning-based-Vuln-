static void pxa2xx_timer_write(void *opaque, target_phys_addr_t offset,

                uint32_t value)

{

    int i, tm = 0;

    PXA2xxTimerInfo *s = (PXA2xxTimerInfo *) opaque;



    switch (offset) {

    case OSMR3:  tm ++;

    case OSMR2:  tm ++;

    case OSMR1:  tm ++;

    case OSMR0:

        s->timer[tm].value = value;

        pxa2xx_timer_update(s, qemu_get_clock(vm_clock));

        break;

    case OSMR11: tm ++;

    case OSMR10: tm ++;

    case OSMR9:  tm ++;

    case OSMR8:  tm ++;

    case OSMR7:  tm ++;

    case OSMR6:  tm ++;

    case OSMR5:  tm ++;

    case OSMR4:

        if (!pxa2xx_timer_has_tm4(s))

            goto badreg;

        s->tm4[tm].tm.value = value;

        pxa2xx_timer_update4(s, qemu_get_clock(vm_clock), tm);

        break;

    case OSCR:

        s->oldclock = s->clock;

        s->lastload = qemu_get_clock(vm_clock);

        s->clock = value;

        pxa2xx_timer_update(s, s->lastload);

        break;

    case OSCR11: tm ++;

    case OSCR10: tm ++;

    case OSCR9:  tm ++;

    case OSCR8:  tm ++;

    case OSCR7:  tm ++;

    case OSCR6:  tm ++;

    case OSCR5:  tm ++;

    case OSCR4:

        if (!pxa2xx_timer_has_tm4(s))

            goto badreg;

        s->tm4[tm].oldclock = s->tm4[tm].clock;

        s->tm4[tm].lastload = qemu_get_clock(vm_clock);

        s->tm4[tm].clock = value;

        pxa2xx_timer_update4(s, s->tm4[tm].lastload, tm);

        break;

    case OIER:

        s->irq_enabled = value & 0xfff;

        break;

    case OSSR:	/* Status register */

        s->events &= ~value;

        for (i = 0; i < 4; i ++, value >>= 1) {

            if (s->timer[i].level && (value & 1)) {

                s->timer[i].level = 0;

                qemu_irq_lower(s->timer[i].irq);

            }

        }

        if (pxa2xx_timer_has_tm4(s)) {

            for (i = 0; i < 8; i ++, value >>= 1)

                if (s->tm4[i].tm.level && (value & 1))

                    s->tm4[i].tm.level = 0;

            if (!(s->events & 0xff0))

                qemu_irq_lower(s->tm4->tm.irq);

        }

        break;

    case OWER:	/* XXX: Reset on OSMR3 match? */

        s->reset3 = value;

        break;

    case OMCR7:  tm ++;

    case OMCR6:  tm ++;

    case OMCR5:  tm ++;

    case OMCR4:

        if (!pxa2xx_timer_has_tm4(s))

            goto badreg;

        s->tm4[tm].control = value & 0x0ff;

        /* XXX Stop if running (shouldn't happen) */

        if ((value & (1 << 7)) || tm == 0)

            s->tm4[tm].freq = pxa2xx_timer4_freq[value & 7];

        else {

            s->tm4[tm].freq = 0;

            pxa2xx_timer_update4(s, qemu_get_clock(vm_clock), tm);

        }

        break;

    case OMCR11: tm ++;

    case OMCR10: tm ++;

    case OMCR9:  tm ++;

    case OMCR8:  tm += 4;

        if (!pxa2xx_timer_has_tm4(s))

            goto badreg;

        s->tm4[tm].control = value & 0x3ff;

        /* XXX Stop if running (shouldn't happen) */

        if ((value & (1 << 7)) || !(tm & 1))

            s->tm4[tm].freq =

                    pxa2xx_timer4_freq[(value & (1 << 8)) ?  0 : (value & 7)];

        else {

            s->tm4[tm].freq = 0;

            pxa2xx_timer_update4(s, qemu_get_clock(vm_clock), tm);

        }

        break;

    default:

    badreg:

        hw_error("pxa2xx_timer_write: Bad offset " REG_FMT "\n", offset);

    }

}
