static void timer_enable(struct xlx_timer *xt)

{

    uint64_t count;



    D(printf("%s timer=%d down=%d\n", __func__,

              xt->nr, xt->regs[R_TCSR] & TCSR_UDT));



    ptimer_stop(xt->ptimer);



    if (xt->regs[R_TCSR] & TCSR_UDT)

        count = xt->regs[R_TLR];

    else

        count = ~0 - xt->regs[R_TLR];

    ptimer_set_count(xt->ptimer, count);

    ptimer_run(xt->ptimer, 1);

}
