static uint64_t g364fb_ctrl_read(void *opaque,

                                 target_phys_addr_t addr,

                                 unsigned int size)

{

    G364State *s = opaque;

    uint32_t val;



    if (addr >= REG_CURS_PAT && addr < REG_CURS_PAT + 0x1000) {

        /* cursor pattern */

        int idx = (addr - REG_CURS_PAT) >> 3;

        val = s->cursor[idx];

    } else if (addr >= REG_CURS_PAL && addr < REG_CURS_PAL + 0x18) {

        /* cursor palette */

        int idx = (addr - REG_CURS_PAL) >> 3;

        val = ((uint32_t)s->cursor_palette[idx][0] << 16);

        val |= ((uint32_t)s->cursor_palette[idx][1] << 8);

        val |= ((uint32_t)s->cursor_palette[idx][2] << 0);

    } else {

        switch (addr) {

            case REG_DISPLAY:

                val = s->width / 4;

                break;

            case REG_VDISPLAY:

                val = s->height * 2;

                break;

            case REG_CTLA:

                val = s->ctla;

                break;

            default:

            {

                error_report("g364: invalid read at [" TARGET_FMT_plx "]",

                             addr);

                val = 0;

                break;

            }

        }

    }



    trace_g364fb_read(addr, val);



    return val;

}
