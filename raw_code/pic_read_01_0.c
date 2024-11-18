static uint64_t pic_read(void *opaque, target_phys_addr_t addr,

                         unsigned size)

{

    HeathrowPICS *s = opaque;

    HeathrowPIC *pic;

    unsigned int n;

    uint32_t value;



    n = ((addr & 0xfff) - 0x10) >> 4;

    if (n >= 2) {

        value = 0;

    } else {

        pic = &s->pics[n];

        switch(addr & 0xf) {

        case 0x0:

            value = pic->events;

            break;

        case 0x4:

            value = pic->mask;

            break;

        case 0xc:

            value = pic->levels;

            break;

        default:

            value = 0;

            break;

        }

    }

    PIC_DPRINTF("readl: " TARGET_FMT_plx " %u: %08x\n", addr, n, value);

    return value;

}
