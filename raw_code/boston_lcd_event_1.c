static void boston_lcd_event(void *opaque, int event)

{

    BostonState *s = opaque;

    if (event == CHR_EVENT_OPENED && !s->lcd_inited) {

        qemu_chr_fe_printf(&s->lcd_display, "        ");

        s->lcd_inited = true;

    }

}
