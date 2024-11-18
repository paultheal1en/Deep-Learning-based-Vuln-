static inline void vmsvga_cursor_define(struct vmsvga_state_s *s,

                struct vmsvga_cursor_definition_s *c)

{

    QEMUCursor *qc;

    int i, pixels;



    qc = cursor_alloc(c->width, c->height);

    qc->hot_x = c->hot_x;

    qc->hot_y = c->hot_y;

    switch (c->bpp) {

    case 1:

        cursor_set_mono(qc, 0xffffff, 0x000000, (void*)c->image,

                        1, (void*)c->mask);

#ifdef DEBUG

        cursor_print_ascii_art(qc, "vmware/mono");

#endif

        break;

    case 32:

        /* fill alpha channel from mask, set color to zero */

        cursor_set_mono(qc, 0x000000, 0x000000, (void*)c->mask,

                        1, (void*)c->mask);

        /* add in rgb values */

        pixels = c->width * c->height;

        for (i = 0; i < pixels; i++) {

            qc->data[i] |= c->image[i] & 0xffffff;

        }

#ifdef DEBUG

        cursor_print_ascii_art(qc, "vmware/32bit");

#endif

        break;

    default:

        fprintf(stderr, "%s: unhandled bpp %d, using fallback cursor\n",

                __FUNCTION__, c->bpp);

        cursor_put(qc);

        qc = cursor_builtin_left_ptr();

    }



    dpy_cursor_define(s->vga.ds, qc);

    cursor_put(qc);

}
