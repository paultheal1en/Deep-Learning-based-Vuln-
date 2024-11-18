static uint64_t pl110_read(void *opaque, hwaddr offset,

                           unsigned size)

{

    pl110_state *s = (pl110_state *)opaque;



    if (offset >= 0xfe0 && offset < 0x1000) {

        return idregs[s->version][(offset - 0xfe0) >> 2];

    }

    if (offset >= 0x200 && offset < 0x400) {

        return s->raw_palette[(offset - 0x200) >> 2];

    }

    switch (offset >> 2) {

    case 0: /* LCDTiming0 */

        return s->timing[0];

    case 1: /* LCDTiming1 */

        return s->timing[1];

    case 2: /* LCDTiming2 */

        return s->timing[2];

    case 3: /* LCDTiming3 */

        return s->timing[3];

    case 4: /* LCDUPBASE */

        return s->upbase;

    case 5: /* LCDLPBASE */

        return s->lpbase;

    case 6: /* LCDIMSC */

        if (s->version != PL110) {

            return s->cr;

        }

        return s->int_mask;

    case 7: /* LCDControl */

        if (s->version != PL110) {

            return s->int_mask;

        }

        return s->cr;

    case 8: /* LCDRIS */

        return s->int_status;

    case 9: /* LCDMIS */

        return s->int_status & s->int_mask;

    case 11: /* LCDUPCURR */

        /* TODO: Implement vertical refresh.  */

        return s->upbase;

    case 12: /* LCDLPCURR */

        return s->lpbase;

    default:

        hw_error("pl110_read: Bad offset %x\n", (int)offset);

        return 0;

    }

}
