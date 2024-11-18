timer_read(void *opaque, target_phys_addr_t addr, unsigned int size)

{

    struct etrax_timer *t = opaque;

    uint32_t r = 0;



    switch (addr) {

    case R_TMR0_DATA:

        r = ptimer_get_count(t->ptimer_t0);

        break;

    case R_TMR1_DATA:

        r = ptimer_get_count(t->ptimer_t1);

        break;

    case R_TIME:

        r = qemu_get_clock_ns(vm_clock) / 10;

        break;

    case RW_INTR_MASK:

        r = t->rw_intr_mask;

        break;

    case R_MASKED_INTR:

        r = t->r_intr & t->rw_intr_mask;

        break;

    default:

        D(printf ("%s %x\n", __func__, addr));

        break;

    }

    return r;

}
