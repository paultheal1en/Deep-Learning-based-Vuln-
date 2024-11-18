static inline void vmsvga_update_rect(struct vmsvga_state_s *s,

                                      int x, int y, int w, int h)

{

    DisplaySurface *surface = qemu_console_surface(s->vga.con);

    int line;

    int bypl;

    int width;

    int start;

    uint8_t *src;

    uint8_t *dst;



    if (x < 0) {

        fprintf(stderr, "%s: update x was < 0 (%d)\n", __func__, x);

        w += x;

        x = 0;

    }

    if (w < 0) {

        fprintf(stderr, "%s: update w was < 0 (%d)\n", __func__, w);

        w = 0;

    }

    if (x + w > surface_width(surface)) {

        fprintf(stderr, "%s: update width too large x: %d, w: %d\n",

                __func__, x, w);

        x = MIN(x, surface_width(surface));

        w = surface_width(surface) - x;

    }



    if (y < 0) {

        fprintf(stderr, "%s: update y was < 0 (%d)\n",  __func__, y);

        h += y;

        y = 0;

    }

    if (h < 0) {

        fprintf(stderr, "%s: update h was < 0 (%d)\n",  __func__, h);

        h = 0;

    }

    if (y + h > surface_height(surface)) {

        fprintf(stderr, "%s: update height too large y: %d, h: %d\n",

                __func__, y, h);

        y = MIN(y, surface_height(surface));

        h = surface_height(surface) - y;

    }



    bypl = surface_stride(surface);

    width = surface_bytes_per_pixel(surface) * w;

    start = surface_bytes_per_pixel(surface) * x + bypl * y;

    src = s->vga.vram_ptr + start;

    dst = surface_data(surface) + start;



    for (line = h; line > 0; line--, src += bypl, dst += bypl) {

        memcpy(dst, src, width);

    }

    dpy_gfx_update(s->vga.con, x, y, w, h);

}
