static void boston_lcd_write(void *opaque, hwaddr addr,

                             uint64_t val, unsigned size)

{

    BostonState *s = opaque;



    switch (size) {

    case 8:

        s->lcd_content[(addr + 7) & 0x7] = val >> 56;

        s->lcd_content[(addr + 6) & 0x7] = val >> 48;

        s->lcd_content[(addr + 5) & 0x7] = val >> 40;

        s->lcd_content[(addr + 4) & 0x7] = val >> 32;

        /* fall through */

    case 4:

        s->lcd_content[(addr + 3) & 0x7] = val >> 24;

        s->lcd_content[(addr + 2) & 0x7] = val >> 16;

        /* fall through */

    case 2:

        s->lcd_content[(addr + 1) & 0x7] = val >> 8;

        /* fall through */

    case 1:

        s->lcd_content[(addr + 0) & 0x7] = val;

        break;

    }



    qemu_chr_fe_printf(&s->lcd_display,

                       "\r%-8.8s", s->lcd_content);

}
