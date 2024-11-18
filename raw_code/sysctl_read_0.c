static uint64_t sysctl_read(void *opaque, target_phys_addr_t addr,

                            unsigned size)

{

    MilkymistSysctlState *s = opaque;

    uint32_t r = 0;



    addr >>= 2;

    switch (addr) {

    case R_TIMER0_COUNTER:

        r = (uint32_t)ptimer_get_count(s->ptimer0);

        /* milkymist timer counts up */

        r = s->regs[R_TIMER0_COMPARE] - r;

        break;

    case R_TIMER1_COUNTER:

        r = (uint32_t)ptimer_get_count(s->ptimer1);

        /* milkymist timer counts up */

        r = s->regs[R_TIMER1_COMPARE] - r;

        break;

    case R_GPIO_IN:

    case R_GPIO_OUT:

    case R_GPIO_INTEN:

    case R_TIMER0_CONTROL:

    case R_TIMER0_COMPARE:

    case R_TIMER1_CONTROL:

    case R_TIMER1_COMPARE:

    case R_ICAP:

    case R_DBG_SCRATCHPAD:

    case R_DBG_WRITE_LOCK:

    case R_CLK_FREQUENCY:

    case R_CAPABILITIES:

    case R_SYSTEM_ID:

        r = s->regs[addr];

        break;



    default:

        error_report("milkymist_sysctl: read access to unknown register 0x"

                TARGET_FMT_plx, addr << 2);

        break;

    }



    trace_milkymist_sysctl_memory_read(addr << 2, r);



    return r;

}
