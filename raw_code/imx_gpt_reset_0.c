static void imx_gpt_reset(DeviceState *dev)

{

    IMXGPTState *s = IMX_GPT(dev);



    /* stop timer */

    ptimer_stop(s->timer);



    /*

     * Soft reset doesn't touch some bits; hard reset clears them

     */

    s->cr &= ~(GPT_CR_EN|GPT_CR_ENMOD|GPT_CR_STOPEN|GPT_CR_DOZEN|

               GPT_CR_WAITEN|GPT_CR_DBGEN);

    s->sr = 0;

    s->pr = 0;

    s->ir = 0;

    s->cnt = 0;

    s->ocr1 = TIMER_MAX;

    s->ocr2 = TIMER_MAX;

    s->ocr3 = TIMER_MAX;

    s->icr1 = 0;

    s->icr2 = 0;



    s->next_timeout = TIMER_MAX;

    s->next_int = 0;



    /* compute new freq */

    imx_gpt_set_freq(s);



    /* reset the limit to TIMER_MAX */

    ptimer_set_limit(s->timer, TIMER_MAX, 1);



    /* if the timer is still enabled, restart it */

    if (s->freq && (s->cr & GPT_CR_EN)) {

        ptimer_run(s->timer, 1);

    }

}
