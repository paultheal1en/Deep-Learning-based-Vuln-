static void imx_timerp_write(void *opaque, target_phys_addr_t offset,

                             uint64_t value, unsigned size)

{

    IMXTimerPState *s = (IMXTimerPState *)opaque;

    DPRINTF("p-write(offset=%x, value = %x)\n", (unsigned int)offset >> 2,

            (unsigned int)value);



    switch (offset >> 2) {

    case 0: /* CR */

        if (value & CR_SWR) {

            imx_timerp_reset(&s->busdev.qdev);

            value &= ~CR_SWR;

        }

        s->cr = value & 0x03ffffff;

        set_timerp_freq(s);



        if (s->freq && (s->cr & CR_EN)) {

            if (!(s->cr & CR_ENMOD)) {

                ptimer_set_count(s->timer, s->lr);

            }

            ptimer_run(s->timer, 0);

        } else {

            ptimer_stop(s->timer);

        }

        break;



    case 1: /* SR - ACK*/

        s->int_level = 0;

        imx_timerp_update(s);

        break;



    case 2: /* LR - set ticks */

        s->lr = value;

        ptimer_set_limit(s->timer, value, !!(s->cr & CR_IOVW));

        break;



    case 3: /* CMP */

        s->cmp = value;

        if (value) {

            IPRINTF(

                "Values for EPIT comparison other than zero not supported\n"

            );

        }

        break;



    default:

        IPRINTF("imx_timerp_write: Bad offset %x\n",

                   (int)offset >> 2);

    }

}
