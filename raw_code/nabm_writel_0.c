static void nabm_writel (void *opaque, uint32_t addr, uint32_t val)

{

    PCIAC97LinkState *d = opaque;

    AC97LinkState *s = &d->ac97;

    AC97BusMasterRegs *r = NULL;

    uint32_t index = addr - s->base[1];

    switch (index) {

    case PI_BDBAR:

    case PO_BDBAR:

    case MC_BDBAR:

        r = &s->bm_regs[GET_BM (index)];

        r->bdbar = val & ~3;

        dolog ("BDBAR[%d] <- %#x (bdbar %#x)\n",

               GET_BM (index), val, r->bdbar);

        break;

    case GLOB_CNT:

        if (val & GC_WR)

            warm_reset (s);

        if (val & GC_CR)

            cold_reset (s);

        if (!(val & (GC_WR | GC_CR)))

            s->glob_cnt = val & GC_VALID_MASK;

        dolog ("glob_cnt <- %#x (glob_cnt %#x)\n", val, s->glob_cnt);

        break;

    case GLOB_STA:

        s->glob_sta &= ~(val & GS_WCLEAR_MASK);

        s->glob_sta |= (val & ~(GS_WCLEAR_MASK | GS_RO_MASK)) & GS_VALID_MASK;

        dolog ("glob_sta <- %#x (glob_sta %#x)\n", val, s->glob_sta);

        break;

    default:

        dolog ("U nabm writel %#x <- %#x\n", addr, val);

        break;

    }

}
