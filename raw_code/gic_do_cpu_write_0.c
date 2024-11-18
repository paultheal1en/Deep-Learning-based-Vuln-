static void gic_do_cpu_write(void *opaque, target_phys_addr_t addr,

                             uint64_t value, unsigned size)

{

    GICState **backref = (GICState **)opaque;

    GICState *s = *backref;

    int id = (backref - s->backref);

    gic_cpu_write(s, id, addr, value);

}
