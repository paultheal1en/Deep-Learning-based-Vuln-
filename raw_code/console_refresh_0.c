static void console_refresh(QemuConsole *s)

{

    DisplaySurface *surface = qemu_console_surface(s);

    TextCell *c;

    int x, y, y1;



    if (s->ds->have_text) {

        s->text_x[0] = 0;

        s->text_y[0] = 0;

        s->text_x[1] = s->width - 1;

        s->text_y[1] = s->height - 1;

        s->cursor_invalidate = 1;

    }



    vga_fill_rect(s, 0, 0, surface_width(surface), surface_height(surface),

                  color_table_rgb[0][COLOR_BLACK]);

    y1 = s->y_displayed;

    for (y = 0; y < s->height; y++) {

        c = s->cells + y1 * s->width;

        for (x = 0; x < s->width; x++) {

            vga_putcharxy(s, x, y, c->ch,

                          &(c->t_attrib));

            c++;

        }

        if (++y1 == s->total_height) {

            y1 = 0;

        }

    }

    console_show_cursor(s, 1);

    dpy_gfx_update(s, 0, 0,

                   surface_width(surface), surface_height(surface));

}
