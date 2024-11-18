static uint32_t nam_readl (void *opaque, uint32_t addr)

{

    PCIAC97LinkState *d = opaque;

    AC97LinkState *s = &d->ac97;

    dolog ("U nam readl %#x\n", addr);

    s->cas = 0;

    return ~0U;

}
