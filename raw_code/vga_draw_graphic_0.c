static void vga_draw_graphic(VGACommonState *s, int full_update)

{

    int y1, y, update, linesize, y_start, double_scan, mask, depth;

    int width, height, shift_control, line_offset, bwidth, bits;

    ram_addr_t page0, page1, page_min, page_max;

    int disp_width, multi_scan, multi_run;

    uint8_t *d;

    uint32_t v, addr1, addr;

    vga_draw_line_func *vga_draw_line;



    full_update |= update_basic_params(s);



    if (!full_update)

        vga_sync_dirty_bitmap(s);



    s->get_resolution(s, &width, &height);

    disp_width = width;



    shift_control = (s->gr[VGA_GFX_MODE] >> 5) & 3;

    double_scan = (s->cr[VGA_CRTC_MAX_SCAN] >> 7);

    if (shift_control != 1) {

        multi_scan = (((s->cr[VGA_CRTC_MAX_SCAN] & 0x1f) + 1) << double_scan)

            - 1;

    } else {

        /* in CGA modes, multi_scan is ignored */

        /* XXX: is it correct ? */

        multi_scan = double_scan;

    }

    multi_run = multi_scan;

    if (shift_control != s->shift_control ||

        double_scan != s->double_scan) {

        full_update = 1;

        s->shift_control = shift_control;

        s->double_scan = double_scan;

    }



    if (shift_control == 0) {

        if (s->sr[VGA_SEQ_CLOCK_MODE] & 8) {

            disp_width <<= 1;

        }

    } else if (shift_control == 1) {

        if (s->sr[VGA_SEQ_CLOCK_MODE] & 8) {

            disp_width <<= 1;

        }

    }



    depth = s->get_bpp(s);

    if (s->line_offset != s->last_line_offset ||

        disp_width != s->last_width ||

        height != s->last_height ||

        s->last_depth != depth) {

#if defined(HOST_WORDS_BIGENDIAN) == defined(TARGET_WORDS_BIGENDIAN)

        if (depth == 16 || depth == 32) {

#else

        if (depth == 32) {

#endif

            qemu_free_displaysurface(s->ds);

            s->ds->surface = qemu_create_displaysurface_from(disp_width, height, depth,

                    s->line_offset,

                    s->vram_ptr + (s->start_addr * 4));

#if defined(HOST_WORDS_BIGENDIAN) != defined(TARGET_WORDS_BIGENDIAN)

            s->ds->surface->pf = qemu_different_endianness_pixelformat(depth);

#endif

            dpy_gfx_resize(s->ds);

        } else {

            qemu_console_resize(s->ds, disp_width, height);

        }

        s->last_scr_width = disp_width;

        s->last_scr_height = height;

        s->last_width = disp_width;

        s->last_height = height;

        s->last_line_offset = s->line_offset;

        s->last_depth = depth;

        full_update = 1;

    } else if (is_buffer_shared(s->ds->surface) &&

               (full_update || ds_get_data(s->ds) != s->vram_ptr

                + (s->start_addr * 4))) {

        qemu_free_displaysurface(s->ds);

        s->ds->surface = qemu_create_displaysurface_from(disp_width,

                height, depth,

                s->line_offset,

                s->vram_ptr + (s->start_addr * 4));

        dpy_gfx_setdata(s->ds);

    }



    s->rgb_to_pixel =

        rgb_to_pixel_dup_table[get_depth_index(s->ds)];



    if (shift_control == 0) {

        full_update |= update_palette16(s);

        if (s->sr[VGA_SEQ_CLOCK_MODE] & 8) {

            v = VGA_DRAW_LINE4D2;

        } else {

            v = VGA_DRAW_LINE4;

        }

        bits = 4;

    } else if (shift_control == 1) {

        full_update |= update_palette16(s);

        if (s->sr[VGA_SEQ_CLOCK_MODE] & 8) {

            v = VGA_DRAW_LINE2D2;

        } else {

            v = VGA_DRAW_LINE2;

        }

        bits = 4;

    } else {

        switch(s->get_bpp(s)) {

        default:

        case 0:

            full_update |= update_palette256(s);

            v = VGA_DRAW_LINE8D2;

            bits = 4;

            break;

        case 8:

            full_update |= update_palette256(s);

            v = VGA_DRAW_LINE8;

            bits = 8;

            break;

        case 15:

            v = VGA_DRAW_LINE15;

            bits = 16;

            break;

        case 16:

            v = VGA_DRAW_LINE16;

            bits = 16;

            break;

        case 24:

            v = VGA_DRAW_LINE24;

            bits = 24;

            break;

        case 32:

            v = VGA_DRAW_LINE32;

            bits = 32;

            break;

        }

    }

    vga_draw_line = vga_draw_line_table[v * NB_DEPTHS + get_depth_index(s->ds)];



    if (!is_buffer_shared(s->ds->surface) && s->cursor_invalidate)

        s->cursor_invalidate(s);



    line_offset = s->line_offset;

#if 0

    printf("w=%d h=%d v=%d line_offset=%d cr[0x09]=0x%02x cr[0x17]=0x%02x linecmp=%d sr[0x01]=0x%02x\n",

           width, height, v, line_offset, s->cr[9], s->cr[VGA_CRTC_MODE],

           s->line_compare, s->sr[VGA_SEQ_CLOCK_MODE]);

#endif

    addr1 = (s->start_addr * 4);

    bwidth = (width * bits + 7) / 8;

    y_start = -1;

    page_min = -1;

    page_max = 0;

    d = ds_get_data(s->ds);

    linesize = ds_get_linesize(s->ds);

    y1 = 0;

    for(y = 0; y < height; y++) {

        addr = addr1;

        if (!(s->cr[VGA_CRTC_MODE] & 1)) {

            int shift;

            /* CGA compatibility handling */

            shift = 14 + ((s->cr[VGA_CRTC_MODE] >> 6) & 1);

            addr = (addr & ~(1 << shift)) | ((y1 & 1) << shift);

        }

        if (!(s->cr[VGA_CRTC_MODE] & 2)) {

            addr = (addr & ~0x8000) | ((y1 & 2) << 14);

        }

        update = full_update;

        page0 = addr;

        page1 = addr + bwidth - 1;

        update |= memory_region_get_dirty(&s->vram, page0, page1 - page0,

                                          DIRTY_MEMORY_VGA);

        /* explicit invalidation for the hardware cursor */

        update |= (s->invalidated_y_table[y >> 5] >> (y & 0x1f)) & 1;

        if (update) {

            if (y_start < 0)

                y_start = y;

            if (page0 < page_min)

                page_min = page0;

            if (page1 > page_max)

                page_max = page1;

            if (!(is_buffer_shared(s->ds->surface))) {

                vga_draw_line(s, d, s->vram_ptr + addr, width);

                if (s->cursor_draw_line)

                    s->cursor_draw_line(s, d, y);

            }

        } else {

            if (y_start >= 0) {

                /* flush to display */

                dpy_gfx_update(s->ds, 0, y_start,

                               disp_width, y - y_start);

                y_start = -1;

            }

        }

        if (!multi_run) {

            mask = (s->cr[VGA_CRTC_MODE] & 3) ^ 3;

            if ((y1 & mask) == mask)

                addr1 += line_offset;

            y1++;

            multi_run = multi_scan;

        } else {

            multi_run--;

        }

        /* line compare acts on the displayed lines */

        if (y == s->line_compare)

            addr1 = 0;

        d += linesize;

    }

    if (y_start >= 0) {

        /* flush to display */

        dpy_gfx_update(s->ds, 0, y_start,

                       disp_width, y - y_start);

    }

    /* reset modified pages */

    if (page_max >= page_min) {

        memory_region_reset_dirty(&s->vram,

                                  page_min,

                                  page_max - page_min,

                                  DIRTY_MEMORY_VGA);

    }

    memset(s->invalidated_y_table, 0, ((height + 31) >> 5) * 4);

}
