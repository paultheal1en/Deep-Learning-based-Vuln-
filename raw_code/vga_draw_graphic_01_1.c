static void vga_draw_graphic(VGACommonState *s, int full_update)

{

    DisplaySurface *surface = qemu_console_surface(s->con);

    int y1, y, update, linesize, y_start, double_scan, mask, depth;

    int width, height, shift_control, line_offset, bwidth, bits;

    ram_addr_t page0, page1;

    DirtyBitmapSnapshot *snap = NULL;

    int disp_width, multi_scan, multi_run;

    uint8_t *d;

    uint32_t v, addr1, addr;

    vga_draw_line_func *vga_draw_line = NULL;

    bool share_surface;

    pixman_format_code_t format;

#ifdef HOST_WORDS_BIGENDIAN

    bool byteswap = !s->big_endian_fb;

#else

    bool byteswap = s->big_endian_fb;

#endif



    full_update |= update_basic_params(s);



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

        if (sr(s, VGA_SEQ_CLOCK_MODE) & 8) {

            disp_width <<= 1;

        }

    } else if (shift_control == 1) {

        if (sr(s, VGA_SEQ_CLOCK_MODE) & 8) {

            disp_width <<= 1;

        }

    }



    depth = s->get_bpp(s);



    /*

     * Check whether we can share the surface with the backend

     * or whether we need a shadow surface. We share native

     * endian surfaces for 15bpp and above and byteswapped

     * surfaces for 24bpp and above.

     */

    format = qemu_default_pixman_format(depth, !byteswap);

    if (format) {

        share_surface = dpy_gfx_check_format(s->con, format)

            && !s->force_shadow;

    } else {

        share_surface = false;

    }

    if (s->line_offset != s->last_line_offset ||

        disp_width != s->last_width ||

        height != s->last_height ||

        s->last_depth != depth ||

        s->last_byteswap != byteswap ||

        share_surface != is_buffer_shared(surface)) {

        if (share_surface) {

            surface = qemu_create_displaysurface_from(disp_width,

                    height, format, s->line_offset,

                    s->vram_ptr + (s->start_addr * 4));

            dpy_gfx_replace_surface(s->con, surface);

        } else {

            qemu_console_resize(s->con, disp_width, height);

            surface = qemu_console_surface(s->con);

        }

        s->last_scr_width = disp_width;

        s->last_scr_height = height;

        s->last_width = disp_width;

        s->last_height = height;

        s->last_line_offset = s->line_offset;

        s->last_depth = depth;

        s->last_byteswap = byteswap;

        full_update = 1;

    } else if (is_buffer_shared(surface) &&

               (full_update || surface_data(surface) != s->vram_ptr

                + (s->start_addr * 4))) {

        pixman_format_code_t format =

            qemu_default_pixman_format(depth, !byteswap);

        surface = qemu_create_displaysurface_from(disp_width,

                height, format, s->line_offset,

                s->vram_ptr + (s->start_addr * 4));

        dpy_gfx_replace_surface(s->con, surface);

    }



    if (shift_control == 0) {

        full_update |= update_palette16(s);

        if (sr(s, VGA_SEQ_CLOCK_MODE) & 8) {

            v = VGA_DRAW_LINE4D2;

        } else {

            v = VGA_DRAW_LINE4;

        }

        bits = 4;

    } else if (shift_control == 1) {

        full_update |= update_palette16(s);

        if (sr(s, VGA_SEQ_CLOCK_MODE) & 8) {

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

            v = s->big_endian_fb ? VGA_DRAW_LINE15_BE : VGA_DRAW_LINE15_LE;

            bits = 16;

            break;

        case 16:

            v = s->big_endian_fb ? VGA_DRAW_LINE16_BE : VGA_DRAW_LINE16_LE;

            bits = 16;

            break;

        case 24:

            v = s->big_endian_fb ? VGA_DRAW_LINE24_BE : VGA_DRAW_LINE24_LE;

            bits = 24;

            break;

        case 32:

            v = s->big_endian_fb ? VGA_DRAW_LINE32_BE : VGA_DRAW_LINE32_LE;

            bits = 32;

            break;

        }

    }

    vga_draw_line = vga_draw_line_table[v];



    if (!is_buffer_shared(surface) && s->cursor_invalidate) {

        s->cursor_invalidate(s);

    }



    line_offset = s->line_offset;

#if 0

    printf("w=%d h=%d v=%d line_offset=%d cr[0x09]=0x%02x cr[0x17]=0x%02x linecmp=%d sr[0x01]=0x%02x\n",

           width, height, v, line_offset, s->cr[9], s->cr[VGA_CRTC_MODE],

           s->line_compare, sr(s, VGA_SEQ_CLOCK_MODE));

#endif

    addr1 = (s->start_addr * 4);

    bwidth = DIV_ROUND_UP(width * bits, 8);

    y_start = -1;

    d = surface_data(surface);

    linesize = surface_stride(surface);

    y1 = 0;



    if (!full_update) {

        ram_addr_t region_start = addr1;

        ram_addr_t region_end = addr1 + line_offset * height;

        vga_sync_dirty_bitmap(s);

        if (s->line_compare < height) {

            /* split screen mode */

            region_start = 0;

        }

        snap = memory_region_snapshot_and_clear_dirty(&s->vram, region_start,

                                                      region_end - region_start,

                                                      DIRTY_MEMORY_VGA);

    }



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

        if (full_update) {

            update = 1;

        } else {

            update = memory_region_snapshot_get_dirty(&s->vram, snap,

                                                      page0, page1 - page0);

        }

        /* explicit invalidation for the hardware cursor (cirrus only) */

        update |= vga_scanline_invalidated(s, y);

        if (update) {

            if (y_start < 0)

                y_start = y;

            if (!(is_buffer_shared(surface))) {

                vga_draw_line(s, d, s->vram_ptr + addr, width);

                if (s->cursor_draw_line)

                    s->cursor_draw_line(s, d, y);

            }

        } else {

            if (y_start >= 0) {

                /* flush to display */

                dpy_gfx_update(s->con, 0, y_start,

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

        dpy_gfx_update(s->con, 0, y_start,

                       disp_width, y - y_start);

    }

    g_free(snap);

    memset(s->invalidated_y_table, 0, sizeof(s->invalidated_y_table));

}
