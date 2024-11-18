static void imx_epit_write(void *opaque, hwaddr offset, uint64_t value,

                           unsigned size)

{

    IMXEPITState *s = IMX_EPIT(opaque);

    uint32_t reg = offset >> 2;

    uint64_t oldcr;



    DPRINTF("(%s, value = 0x%08x)\n", imx_epit_reg_name(reg), (uint32_t)value);



    switch (reg) {

    case 0: /* CR */



        oldcr = s->cr;

        s->cr = value & 0x03ffffff;

        if (s->cr & CR_SWR) {

            /* handle the reset */

            imx_epit_reset(DEVICE(s));

        } else {

            imx_epit_set_freq(s);

        }



        if (s->freq && (s->cr & CR_EN) && !(oldcr & CR_EN)) {

            if (s->cr & CR_ENMOD) {

                if (s->cr & CR_RLD) {

                    ptimer_set_limit(s->timer_reload, s->lr, 1);

                    ptimer_set_limit(s->timer_cmp, s->lr, 1);

                } else {

                    ptimer_set_limit(s->timer_reload, TIMER_MAX, 1);

                    ptimer_set_limit(s->timer_cmp, TIMER_MAX, 1);

                }

            }



            imx_epit_reload_compare_timer(s);

            ptimer_run(s->timer_reload, 0);

            if (s->cr & CR_OCIEN) {

                ptimer_run(s->timer_cmp, 0);

            } else {

                ptimer_stop(s->timer_cmp);

            }

        } else if (!(s->cr & CR_EN)) {

            /* stop both timers */

            ptimer_stop(s->timer_reload);

            ptimer_stop(s->timer_cmp);

        } else  if (s->cr & CR_OCIEN) {

            if (!(oldcr & CR_OCIEN)) {

                imx_epit_reload_compare_timer(s);

                ptimer_run(s->timer_cmp, 0);

            }

        } else {

            ptimer_stop(s->timer_cmp);

        }

        break;



    case 1: /* SR - ACK*/

        /* writing 1 to OCIF clear the OCIF bit */

        if (value & 0x01) {

            s->sr = 0;

            imx_epit_update_int(s);

        }

        break;



    case 2: /* LR - set ticks */

        s->lr = value;



        if (s->cr & CR_RLD) {

            /* Also set the limit if the LRD bit is set */

            /* If IOVW bit is set then set the timer value */

            ptimer_set_limit(s->timer_reload, s->lr, s->cr & CR_IOVW);

            ptimer_set_limit(s->timer_cmp, s->lr, 0);

        } else if (s->cr & CR_IOVW) {

            /* If IOVW bit is set then set the timer value */

            ptimer_set_count(s->timer_reload, s->lr);

        }



        imx_epit_reload_compare_timer(s);

        break;



    case 3: /* CMP */

        s->cmp = value;



        imx_epit_reload_compare_timer(s);



        break;



    default:

        IPRINTF("Bad offset %x\n", reg);



        break;

    }

}
