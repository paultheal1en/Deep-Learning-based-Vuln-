static inline void vmsvga_update_rect(struct vmsvga_state_s *s,

                int x, int y, int w, int h)

{

#ifndef DIRECT_VRAM

    int line = h;

    int bypl = s->bypp * s->width;

    int width = s->bypp * w;

    int start = s->bypp * x + bypl * y;

    uint8_t *src = s->vram + start;

    uint8_t *dst = s->ds->data + start;



    for (; line > 0; line --, src += bypl, dst += bypl)

        memcpy(dst, src, width);

#endif



    dpy_update(s->ds, x, y, w, h);

}
