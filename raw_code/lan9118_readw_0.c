static uint32_t lan9118_readw(void *opaque, target_phys_addr_t offset)

{

    lan9118_state *s = (lan9118_state *)opaque;

    uint32_t val;



    if (s->read_word_prev_offset != (offset & ~0x3)) {

        /* New offset, reset word counter */

        s->read_word_n = 0;

        s->read_word_prev_offset = offset & ~0x3;

    }



    s->read_word_n++;

    if (s->read_word_n == 1) {

        s->read_long = lan9118_readl(s, offset & ~3, 4);

    } else {

        s->read_word_n = 0;

    }



    if (offset & 2) {

        val = s->read_long >> 16;

    } else {

        val = s->read_long & 0xFFFF;

    }



    //DPRINTF("Readw reg 0x%02x, val 0x%x\n", (int)offset, val);

    return val;

}
