static void nabm_writeb (void *opaque, uint32_t addr, uint32_t val)

{

    PCIAC97LinkState *d = opaque;

    AC97LinkState *s = &d->ac97;

    AC97BusMasterRegs *r = NULL;

    uint32_t index = addr - s->base[1];

    switch (index) {

    case PI_LVI:

    case PO_LVI:

    case MC_LVI:

        r = &s->bm_regs[GET_BM (index)];

        if ((r->cr & CR_RPBM) && (r->sr & SR_DCH)) {

            r->sr &= ~(SR_DCH | SR_CELV);

            r->civ = r->piv;

            r->piv = (r->piv + 1) % 32;

            fetch_bd (s, r);

        }

        r->lvi = val % 32;

        dolog ("LVI[%d] <- %#x\n", GET_BM (index), val);

        break;

    case PI_CR:

    case PO_CR:

    case MC_CR:

        r = &s->bm_regs[GET_BM (index)];

        if (val & CR_RR) {

            reset_bm_regs (s, r);

        }

        else {

            r->cr = val & CR_VALID_MASK;

            if (!(r->cr & CR_RPBM)) {

                voice_set_active (s, r - s->bm_regs, 0);

                r->sr |= SR_DCH;

            }

            else {

                r->civ = r->piv;

                r->piv = (r->piv + 1) % 32;

                fetch_bd (s, r);

                r->sr &= ~SR_DCH;

                voice_set_active (s, r - s->bm_regs, 1);

            }

        }

        dolog ("CR[%d] <- %#x (cr %#x)\n", GET_BM (index), val, r->cr);

        break;

    case PI_SR:

    case PO_SR:

    case MC_SR:

        r = &s->bm_regs[GET_BM (index)];

        r->sr |= val & ~(SR_RO_MASK | SR_WCLEAR_MASK);

        update_sr (s, r, r->sr & ~(val & SR_WCLEAR_MASK));

        dolog ("SR[%d] <- %#x (sr %#x)\n", GET_BM (index), val, r->sr);

        break;

    default:

        dolog ("U nabm writeb %#x <- %#x\n", addr, val);

        break;

    }

}
