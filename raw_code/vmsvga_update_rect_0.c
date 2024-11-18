static inline void vmsvga_update_rect(struct vmsvga_state_s *s,

                int x, int y, int w, int h)

{

    int line;

    int bypl;

    int width;

    int start;

    uint8_t *src;

    uint8_t *dst;



    if (x + w > s->width) {

        fprintf(stderr, "%s: update width too large x: %d, w: %d\n",

                        __FUNCTION__, x, w);

        x = MIN(x, s->width);

        w = s->width - x;

    }



    if (y + h > s->height) {

        fprintf(stderr, "%s: update height too large y: %d, h: %d\n",

                        __FUNCTION__, y, h);

        y = MIN(y, s->height);

        h = s->height - y;

    }



    line = h;

    bypl = s->bypp * s->width;

    width = s->bypp * w;

    start = s->bypp * x + bypl * y;

    src = s->vga.vram_ptr + start;

    dst = ds_get_data(s->vga.ds) + start;



    for (; line > 0; line --, src += bypl, dst += bypl)

        memcpy(dst, src, width);



    dpy_gfx_update(s->vga.ds, x, y, w, h);

}
