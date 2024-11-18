static void gic_dist_writel(void *opaque, target_phys_addr_t offset,

                            uint32_t value)

{

    GICState *s = (GICState *)opaque;

    if (offset == 0xf00) {

        int cpu;

        int irq;

        int mask;



        cpu = gic_get_current_cpu(s);

        irq = value & 0x3ff;

        switch ((value >> 24) & 3) {

        case 0:

            mask = (value >> 16) & ALL_CPU_MASK;

            break;

        case 1:

            mask = ALL_CPU_MASK ^ (1 << cpu);

            break;

        case 2:

            mask = 1 << cpu;

            break;

        default:

            DPRINTF("Bad Soft Int target filter\n");

            mask = ALL_CPU_MASK;

            break;

        }

        GIC_SET_PENDING(irq, mask);

        gic_update(s);

        return;

    }

    gic_dist_writew(opaque, offset, value & 0xffff);

    gic_dist_writew(opaque, offset + 2, value >> 16);

}
