static uint32_t grlib_gptimer_readl(void *opaque, target_phys_addr_t addr)

{

    GPTimerUnit        *unit  = opaque;

    target_phys_addr_t  timer_addr;

    int                 id;

    uint32_t            value = 0;



    addr &= 0xff;



    /* Unit registers */

    switch (addr) {

    case SCALER_OFFSET:

        trace_grlib_gptimer_readl(-1, "scaler:", unit->scaler);

        return unit->scaler;



    case SCALER_RELOAD_OFFSET:

        trace_grlib_gptimer_readl(-1, "reload:", unit->reload);

        return unit->reload;



    case CONFIG_OFFSET:

        trace_grlib_gptimer_readl(-1, "config:", unit->config);

        return unit->config;



    default:

        break;

    }



    timer_addr = (addr % TIMER_BASE);

    id         = (addr - TIMER_BASE) / TIMER_BASE;



    if (id >= 0 && id < unit->nr_timers) {



        /* GPTimer registers */

        switch (timer_addr) {

        case COUNTER_OFFSET:

            value = ptimer_get_count(unit->timers[id].ptimer);

            trace_grlib_gptimer_readl(id, "counter value:", value);

            return value;



        case COUNTER_RELOAD_OFFSET:

            value = unit->timers[id].reload;

            trace_grlib_gptimer_readl(id, "reload value:", value);

            return value;



        case CONFIG_OFFSET:

            trace_grlib_gptimer_readl(id, "scaler value:",

                                      unit->timers[id].config);

            return unit->timers[id].config;



        default:

            break;

        }



    }



    trace_grlib_gptimer_unknown_register("read", addr);

    return 0;

}
