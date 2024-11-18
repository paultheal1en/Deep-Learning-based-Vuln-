static void imx_gpt_write(void *opaque, hwaddr offset, uint64_t value,

                          unsigned size)

{

    IMXGPTState *s = IMX_GPT(opaque);

    uint32_t oldreg;

    uint32_t reg = offset >> 2;



    DPRINTF("(%s, value = 0x%08x)\n", imx_gpt_reg_name(reg),

            (uint32_t)value);



    switch (reg) {

    case 0:

        oldreg = s->cr;

        s->cr = value & ~0x7c14;

        if (s->cr & GPT_CR_SWR) { /* force reset */

            /* handle the reset */

            imx_gpt_reset(DEVICE(s));

        } else {

            /* set our freq, as the source might have changed */

            imx_gpt_set_freq(s);



            if ((oldreg ^ s->cr) & GPT_CR_EN) {

                if (s->cr & GPT_CR_EN) {

                    if (s->cr & GPT_CR_ENMOD) {

                        s->next_timeout = TIMER_MAX;

                        ptimer_set_count(s->timer, TIMER_MAX);

                        imx_gpt_compute_next_timeout(s, false);

                    }

                    ptimer_run(s->timer, 1);

                } else {

                    /* stop timer */

                    ptimer_stop(s->timer);

                }

            }

        }

        break;



    case 1: /* Prescaler */

        s->pr = value & 0xfff;

        imx_gpt_set_freq(s);

        break;



    case 2: /* SR */

        s->sr &= ~(value & 0x3f);

        imx_gpt_update_int(s);

        break;



    case 3: /* IR -- interrupt register */

        s->ir = value & 0x3f;

        imx_gpt_update_int(s);



        imx_gpt_compute_next_timeout(s, false);



        break;



    case 4: /* OCR1 -- output compare register */

        s->ocr1 = value;



        /* In non-freerun mode, reset count when this register is written */

        if (!(s->cr & GPT_CR_FRR)) {

            s->next_timeout = TIMER_MAX;

            ptimer_set_limit(s->timer, TIMER_MAX, 1);

        }



        /* compute the new timeout */

        imx_gpt_compute_next_timeout(s, false);



        break;



    case 5: /* OCR2 -- output compare register */

        s->ocr2 = value;



        /* compute the new timeout */

        imx_gpt_compute_next_timeout(s, false);



        break;



    case 6: /* OCR3 -- output compare register */

        s->ocr3 = value;



        /* compute the new timeout */

        imx_gpt_compute_next_timeout(s, false);



        break;



    default:

        IPRINTF("Bad offset %x\n", reg);

        break;

    }

}
