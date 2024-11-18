static void omap_update_display(void *opaque)

{

    struct omap_lcd_panel_s *omap_lcd = (struct omap_lcd_panel_s *) opaque;

    DisplaySurface *surface = qemu_console_surface(omap_lcd->con);

    draw_line_func draw_line;

    int size, height, first, last;

    int width, linesize, step, bpp, frame_offset;

    hwaddr frame_base;



    if (!omap_lcd || omap_lcd->plm == 1 || !omap_lcd->enable ||

        !surface_bits_per_pixel(surface)) {

        return;

    }



    frame_offset = 0;

    if (omap_lcd->plm != 2) {

        cpu_physical_memory_read(omap_lcd->dma->phys_framebuffer[

                                  omap_lcd->dma->current_frame],

                                 (void *)omap_lcd->palette, 0x200);

        switch (omap_lcd->palette[0] >> 12 & 7) {

        case 3 ... 7:

            frame_offset += 0x200;

            break;

        default:

            frame_offset += 0x20;

        }

    }



    /* Colour depth */

    switch ((omap_lcd->palette[0] >> 12) & 7) {

    case 1:

        draw_line = draw_line_table2[surface_bits_per_pixel(surface)];

        bpp = 2;

        break;



    case 2:

        draw_line = draw_line_table4[surface_bits_per_pixel(surface)];

        bpp = 4;

        break;



    case 3:

        draw_line = draw_line_table8[surface_bits_per_pixel(surface)];

        bpp = 8;

        break;



    case 4 ... 7:

        if (!omap_lcd->tft)

            draw_line = draw_line_table12[surface_bits_per_pixel(surface)];

        else

            draw_line = draw_line_table16[surface_bits_per_pixel(surface)];

        bpp = 16;

        break;



    default:

        /* Unsupported at the moment.  */

        return;

    }



    /* Resolution */

    width = omap_lcd->width;

    if (width != surface_width(surface) ||

        omap_lcd->height != surface_height(surface)) {

        qemu_console_resize(omap_lcd->con,

                            omap_lcd->width, omap_lcd->height);

        surface = qemu_console_surface(omap_lcd->con);

        omap_lcd->invalidate = 1;

    }



    if (omap_lcd->dma->current_frame == 0)

        size = omap_lcd->dma->src_f1_bottom - omap_lcd->dma->src_f1_top;

    else

        size = omap_lcd->dma->src_f2_bottom - omap_lcd->dma->src_f2_top;



    if (frame_offset + ((width * omap_lcd->height * bpp) >> 3) > size + 2) {

        omap_lcd->sync_error = 1;

        omap_lcd_interrupts(omap_lcd);

        omap_lcd->enable = 0;

        return;

    }



    /* Content */

    frame_base = omap_lcd->dma->phys_framebuffer[

            omap_lcd->dma->current_frame] + frame_offset;

    omap_lcd->dma->condition |= 1 << omap_lcd->dma->current_frame;

    if (omap_lcd->dma->interrupts & 1)

        qemu_irq_raise(omap_lcd->dma->irq);

    if (omap_lcd->dma->dual)

        omap_lcd->dma->current_frame ^= 1;



    if (!surface_bits_per_pixel(surface)) {

        return;

    }



    first = 0;

    height = omap_lcd->height;

    if (omap_lcd->subpanel & (1 << 31)) {

        if (omap_lcd->subpanel & (1 << 29))

            first = (omap_lcd->subpanel >> 16) & 0x3ff;

        else

            height = (omap_lcd->subpanel >> 16) & 0x3ff;

        /* TODO: fill the rest of the panel with DPD */

    }



    step = width * bpp >> 3;

    linesize = surface_stride(surface);

    framebuffer_update_display(surface, omap_lcd->sysmem,

                               frame_base, width, height,

                               step, linesize, 0,

                               omap_lcd->invalidate,

                               draw_line, omap_lcd->palette,

                               &first, &last);

    if (first >= 0) {

        dpy_gfx_update(omap_lcd->con, 0, first, width, last - first + 1);

    }

    omap_lcd->invalidate = 0;

}
