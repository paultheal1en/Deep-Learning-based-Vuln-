void console_select(unsigned int index)

{

    TextConsole *s;



    if (index >= MAX_CONSOLES)

        return;

    if (active_console) {

        active_console->g_width = ds_get_width(active_console->ds);

        active_console->g_height = ds_get_height(active_console->ds);

    }

    s = consoles[index];

    if (s) {

        DisplayState *ds = s->ds;



        if (active_console->cursor_timer) {

            qemu_del_timer(active_console->cursor_timer);

        }

        active_console = s;

        if (ds_get_bits_per_pixel(s->ds)) {

            ds->surface = qemu_resize_displaysurface(ds, s->g_width, s->g_height);

        } else {

            s->ds->surface->width = s->width;

            s->ds->surface->height = s->height;

        }

        if (s->cursor_timer) {

            qemu_mod_timer(s->cursor_timer,

                   qemu_get_clock_ms(rt_clock) + CONSOLE_CURSOR_PERIOD / 2);

        }

        dpy_resize(s->ds);

        vga_hw_invalidate();

    }

}
