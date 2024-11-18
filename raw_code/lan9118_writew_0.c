static void lan9118_writew(void *opaque, target_phys_addr_t offset,

                           uint32_t val)

{

    lan9118_state *s = (lan9118_state *)opaque;

    offset &= 0xff;



    if (s->write_word_prev_offset != (offset & ~0x3)) {

        /* New offset, reset word counter */

        s->write_word_n = 0;

        s->write_word_prev_offset = offset & ~0x3;

    }



    if (offset & 0x2) {

        s->write_word_h = val;

    } else {

        s->write_word_l = val;

    }



    //DPRINTF("Writew reg 0x%02x = 0x%08x\n", (int)offset, val);

    s->write_word_n++;

    if (s->write_word_n == 2) {

        s->write_word_n = 0;

        lan9118_writel(s, offset & ~3, s->write_word_l +

                (s->write_word_h << 16), 4);

    }

}
