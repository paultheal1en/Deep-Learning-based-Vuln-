void console_select(unsigned int index)

{

    TextConsole *s;



    if (index >= MAX_CONSOLES)

        return;

    s = consoles[index];

    if (s) {

        active_console = s;

        if (s->g_width && s->g_height

            && (s->g_width != s->ds->width || s->g_height != s->ds->height))

            dpy_resize(s->ds, s->g_width, s->g_height);

        vga_hw_invalidate();

    }

}
