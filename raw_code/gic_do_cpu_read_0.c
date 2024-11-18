static uint64_t gic_do_cpu_read(void *opaque, target_phys_addr_t addr,

                                unsigned size)

{

    GICState **backref = (GICState **)opaque;

    GICState *s = *backref;

    int id = (backref - s->backref);

    return gic_cpu_read(s, id, addr);

}
