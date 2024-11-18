static DisplaySurface *qemu_create_message_surface(int w, int h,

                                                   const char *msg)

{

    DisplaySurface *surface = qemu_create_displaysurface(w, h);

    pixman_color_t bg = color_table_rgb[0][COLOR_BLACK];

    pixman_color_t fg = color_table_rgb[0][COLOR_WHITE];

    pixman_image_t *glyph;

    int len, x, y, i;



    len = strlen(msg);

    x = (w / FONT_WIDTH  - len) / 2;

    y = (h / FONT_HEIGHT - 1)   / 2;

    for (i = 0; i < len; i++) {

        glyph = qemu_pixman_glyph_from_vgafont(FONT_HEIGHT, vgafont16, msg[i]);

        qemu_pixman_glyph_render(glyph, surface->image, &fg, &bg,

                                 x+i, y, FONT_WIDTH, FONT_HEIGHT);

        qemu_pixman_image_unref(glyph);

    }

    return surface;

}
