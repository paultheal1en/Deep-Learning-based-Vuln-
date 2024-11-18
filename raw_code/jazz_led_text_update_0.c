static void jazz_led_text_update(void *opaque, console_ch_t *chardata)

{

    LedState *s = opaque;

    char buf[2];



    dpy_text_cursor(s->con, -1, -1);

    qemu_console_resize(s->con, 2, 1);



    /* TODO: draw the segments */

    snprintf(buf, 2, "%02hhx\n", s->segments);

    console_write_ch(chardata++, 0x00200100 | buf[0]);

    console_write_ch(chardata++, 0x00200100 | buf[1]);



    dpy_text_update(s->con, 0, 0, 2, 1);

}
