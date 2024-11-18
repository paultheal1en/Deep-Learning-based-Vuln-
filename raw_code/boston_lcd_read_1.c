static uint64_t boston_lcd_read(void *opaque, hwaddr addr,

                                unsigned size)

{

    BostonState *s = opaque;

    uint64_t val = 0;



    switch (size) {

    case 8:

        val |= (uint64_t)s->lcd_content[(addr + 7) & 0x7] << 56;

        val |= (uint64_t)s->lcd_content[(addr + 6) & 0x7] << 48;

        val |= (uint64_t)s->lcd_content[(addr + 5) & 0x7] << 40;

        val |= (uint64_t)s->lcd_content[(addr + 4) & 0x7] << 32;

        /* fall through */

    case 4:

        val |= (uint64_t)s->lcd_content[(addr + 3) & 0x7] << 24;

        val |= (uint64_t)s->lcd_content[(addr + 2) & 0x7] << 16;

        /* fall through */

    case 2:

        val |= (uint64_t)s->lcd_content[(addr + 1) & 0x7] << 8;

        /* fall through */

    case 1:

        val |= (uint64_t)s->lcd_content[(addr + 0) & 0x7];

        break;

    }



    return val;

}
