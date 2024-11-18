ioapic_mem_write(void *opaque, target_phys_addr_t addr, uint64_t val,

                 unsigned int size)

{

    IOAPICCommonState *s = opaque;

    int index;



    switch (addr & 0xff) {

    case IOAPIC_IOREGSEL:

        s->ioregsel = val;

        break;

    case IOAPIC_IOWIN:

        if (size != 4) {

            break;

        }

        DPRINTF("write: %08x = %08" PRIx64 "\n", s->ioregsel, val);

        switch (s->ioregsel) {

        case IOAPIC_REG_ID:

            s->id = (val >> IOAPIC_ID_SHIFT) & IOAPIC_ID_MASK;

            break;

        case IOAPIC_REG_VER:

        case IOAPIC_REG_ARB:

            break;

        default:

            index = (s->ioregsel - IOAPIC_REG_REDTBL_BASE) >> 1;

            if (index >= 0 && index < IOAPIC_NUM_PINS) {

                if (s->ioregsel & 1) {

                    s->ioredtbl[index] &= 0xffffffff;

                    s->ioredtbl[index] |= (uint64_t)val << 32;

                } else {

                    s->ioredtbl[index] &= ~0xffffffffULL;

                    s->ioredtbl[index] |= val;

                }

                ioapic_service(s);

            }

        }

        break;

    }

}
