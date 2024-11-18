static void nabm_writew (void *opaque, uint32_t addr, uint32_t val)

{

    PCIAC97LinkState *d = opaque;

    AC97LinkState *s = &d->ac97;

    AC97BusMasterRegs *r = NULL;

    uint32_t index = addr - s->base[1];

    switch (index) {

    case PI_SR:

    case PO_SR:

    case MC_SR:

        r = &s->bm_regs[GET_BM (index)];

        r->sr |= val & ~(SR_RO_MASK | SR_WCLEAR_MASK);

        update_sr (s, r, r->sr & ~(val & SR_WCLEAR_MASK));

        dolog ("SR[%d] <- %#x (sr %#x)\n", GET_BM (index), val, r->sr);

        break;

    default:

        dolog ("U nabm writew %#x <- %#x\n", addr, val);

        break;

    }

}
