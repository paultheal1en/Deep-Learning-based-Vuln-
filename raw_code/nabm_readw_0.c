static uint32_t nabm_readw (void *opaque, uint32_t addr)

{

    PCIAC97LinkState *d = opaque;

    AC97LinkState *s = &d->ac97;

    AC97BusMasterRegs *r = NULL;

    uint32_t index = addr - s->base[1];

    uint32_t val = ~0U;



    switch (index) {

    case PI_SR:

    case PO_SR:

    case MC_SR:

        r = &s->bm_regs[GET_BM (index)];

        val = r->sr;

        dolog ("SR[%d] -> %#x\n", GET_BM (index), val);

        break;

    case PI_PICB:

    case PO_PICB:

    case MC_PICB:

        r = &s->bm_regs[GET_BM (index)];

        val = r->picb;

        dolog ("PICB[%d] -> %#x\n", GET_BM (index), val);

        break;

    default:

        dolog ("U nabm readw %#x -> %#x\n", addr, val);

        break;

    }

    return val;

}
