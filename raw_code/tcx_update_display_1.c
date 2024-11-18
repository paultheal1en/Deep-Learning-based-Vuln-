static void tcx_update_display(void *opaque)

{

    TCXState *ts = opaque;

    ram_addr_t page, page_min, page_max;

    int y, y_start, dd, ds;

    uint8_t *d, *s;

    void (*f)(TCXState *s1, uint8_t *dst, const uint8_t *src, int width);



    if (ts->ds->depth == 0)

        return;



    page = ts->vram_offset;

    y_start = -1;

    page_min = 0xffffffff;

    page_max = 0;

    d = ts->ds->data;

    s = ts->vram;

    dd = ts->ds->linesize;

    ds = 1024;



    switch (ts->ds->depth) {

    case 32:

        f = tcx_draw_line32;

        break;

    case 15:

    case 16:

        f = tcx_draw_line16;

        break;

    default:

    case 8:

        f = tcx_draw_line8;

        break;

    case 0:

        return;

    }



    for(y = 0; y < ts->height; y += 4, page += TARGET_PAGE_SIZE) {

        if (cpu_physical_memory_get_dirty(page, VGA_DIRTY_FLAG)) {

            if (y_start < 0)

                y_start = y;

            if (page < page_min)

                page_min = page;

            if (page > page_max)

                page_max = page;

            f(ts, d, s, ts->width);

            d += dd;

            s += ds;

            f(ts, d, s, ts->width);

            d += dd;

            s += ds;

            f(ts, d, s, ts->width);

            d += dd;

            s += ds;

            f(ts, d, s, ts->width);

            d += dd;

            s += ds;

        } else {

            if (y_start >= 0) {

                /* flush to display */

                dpy_update(ts->ds, 0, y_start,

                           ts->width, y - y_start);

                y_start = -1;

            }

            d += dd * 4;

            s += ds * 4;

        }

    }

    if (y_start >= 0) {

        /* flush to display */

        dpy_update(ts->ds, 0, y_start,

                   ts->width, y - y_start);

    }

    /* reset modified pages */

    if (page_min <= page_max) {

        cpu_physical_memory_reset_dirty(page_min, page_max + TARGET_PAGE_SIZE,

                                        VGA_DIRTY_FLAG);

    }

}