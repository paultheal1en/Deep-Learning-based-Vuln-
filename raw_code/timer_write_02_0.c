timer_write(void *opaque, hwaddr addr,

            uint64_t val64, unsigned int size)

{

    struct timerblock *t = opaque;

    struct xlx_timer *xt;

    unsigned int timer;

    uint32_t value = val64;



    addr >>= 2;

    timer = timer_from_addr(addr);

    xt = &t->timers[timer];

    D(fprintf(stderr, "%s addr=%x val=%x (timer=%d off=%d)\n",

             __func__, addr * 4, value, timer, addr & 3));

    /* Further decoding to address a specific timers reg.  */

    addr &= 3;

    switch (addr) 

    {

        case R_TCSR:

            if (value & TCSR_TINT)

                value &= ~TCSR_TINT;



            xt->regs[addr] = value;

            if (value & TCSR_ENT)

                timer_enable(xt);

            break;

 

        default:

            if (addr < ARRAY_SIZE(xt->regs))

                xt->regs[addr] = value;

            break;

    }

    timer_update_irq(t);

}
