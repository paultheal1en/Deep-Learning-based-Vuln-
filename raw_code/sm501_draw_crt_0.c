static void sm501_draw_crt(SM501State *s)

{

    DisplaySurface *surface = qemu_console_surface(s->con);

    int y, c_x = 0, c_y = 0;

    uint8_t *hwc_src = NULL, *src = s->local_mem;

    int width = get_width(s, 1);

    int height = get_height(s, 1);

    int src_bpp = get_bpp(s, 1);

    int dst_bpp = surface_bytes_per_pixel(surface);

    uint32_t *palette = (uint32_t *)&s->dc_palette[SM501_DC_CRT_PALETTE -

                                                   SM501_DC_PANEL_PALETTE];

    uint8_t hwc_palette[3 * 3];

    int ds_depth_index = get_depth_index(surface);

    draw_line_func *draw_line = NULL;

    draw_hwc_line_func *draw_hwc_line = NULL;

    int full_update = 0;

    int y_start = -1;

    ram_addr_t page_min = ~0l;

    ram_addr_t page_max = 0l;

    ram_addr_t offset = 0;



    /* choose draw_line function */

    switch (src_bpp) {

    case 1:

        draw_line = draw_line8_funcs[ds_depth_index];

        break;

    case 2:

        draw_line = draw_line16_funcs[ds_depth_index];

        break;

    case 4:

        draw_line = draw_line32_funcs[ds_depth_index];

        break;

    default:

        printf("sm501 draw crt : invalid DC_CRT_CONTROL=%x.\n",

               s->dc_crt_control);

        abort();

        break;

    }



    /* set up to draw hardware cursor */

    if (is_hwc_enabled(s, 1)) {

        /* choose cursor draw line function */

        draw_hwc_line = draw_hwc_line_funcs[ds_depth_index];

        hwc_src = get_hwc_address(s, 1);

        c_x = get_hwc_x(s, 1);

        c_y = get_hwc_y(s, 1);

        get_hwc_palette(s, 1, hwc_palette);

    }



    /* adjust console size */

    if (s->last_width != width || s->last_height != height) {

        qemu_console_resize(s->con, width, height);

        surface = qemu_console_surface(s->con);

        s->last_width = width;

        s->last_height = height;

        full_update = 1;

    }



    /* draw each line according to conditions */

    memory_region_sync_dirty_bitmap(&s->local_mem_region);

    for (y = 0; y < height; y++) {

        int update, update_hwc;

        ram_addr_t page0 = offset;

        ram_addr_t page1 = offset + width * src_bpp - 1;



        /* check if hardware cursor is enabled and we're within its range */

        update_hwc = draw_hwc_line && c_y <= y && y < c_y + SM501_HWC_HEIGHT;

        update = full_update || update_hwc;

        /* check dirty flags for each line */

        update |= memory_region_get_dirty(&s->local_mem_region, page0,

                                          page1 - page0, DIRTY_MEMORY_VGA);



        /* draw line and change status */

        if (update) {

            uint8_t *d = surface_data(surface);

            d +=  y * width * dst_bpp;



            /* draw graphics layer */

            draw_line(d, src, width, palette);



            /* draw hardware cursor */

            if (update_hwc) {

                draw_hwc_line(d, hwc_src, width, hwc_palette, c_x, y - c_y);

            }



            if (y_start < 0) {

                y_start = y;

            }

            if (page0 < page_min) {

                page_min = page0;

            }

            if (page1 > page_max) {

                page_max = page1;

            }

        } else {

            if (y_start >= 0) {

                /* flush to display */

                dpy_gfx_update(s->con, 0, y_start, width, y - y_start);

                y_start = -1;

            }

        }



        src += width * src_bpp;

        offset += width * src_bpp;

    }



    /* complete flush to display */

    if (y_start >= 0) {

        dpy_gfx_update(s->con, 0, y_start, width, y - y_start);

    }



    /* clear dirty flags */

    if (page_min != ~0l) {

        memory_region_reset_dirty(&s->local_mem_region,

                                  page_min, page_max + TARGET_PAGE_SIZE,

                                  DIRTY_MEMORY_VGA);

    }

}
