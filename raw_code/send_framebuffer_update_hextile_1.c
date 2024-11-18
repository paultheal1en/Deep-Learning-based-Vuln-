static void send_framebuffer_update_hextile(VncState *vs, int x, int y, int w, int h)

{

    int i, j;

    int has_fg, has_bg;

    uint8_t *last_fg, *last_bg;



    vnc_framebuffer_update(vs, x, y, w, h, 5);



    last_fg = (uint8_t *) malloc(vs->depth);

    last_bg = (uint8_t *) malloc(vs->depth);

    has_fg = has_bg = 0;

    for (j = y; j < (y + h); j += 16) {

	for (i = x; i < (x + w); i += 16) {

            vs->send_hextile_tile(vs, i, j,

                                  MIN(16, x + w - i), MIN(16, y + h - j),

                                  last_bg, last_fg, &has_bg, &has_fg);

	}

    }

    free(last_fg);

    free(last_bg);



}
