static void pxa2xx_lcdc_dma0_redraw_vert(struct pxa2xx_lcdc_s *s,

                uint8_t *fb, int *miny, int *maxy)

{

    int y, src_width, dest_width, dirty[2];

    uint8_t *src, *dest;

    ram_addr_t x, addr, new_addr, start, end;

    drawfn fn = 0;

    if (s->dest_width)

        fn = s->line_fn[s->transp][s->bpp];

    if (!fn)

        return;



    src = fb;

    src_width = (s->xres + 3) & ~3;     /* Pad to a 4 pixels multiple */

    if (s->bpp == pxa_lcdc_19pbpp || s->bpp == pxa_lcdc_18pbpp)

        src_width *= 3;

    else if (s->bpp > pxa_lcdc_16bpp)

        src_width *= 4;

    else if (s->bpp > pxa_lcdc_8bpp)

        src_width *= 2;



    dest_width = s->yres * s->dest_width;

    dest = ds_get_data(s->ds) + dest_width * (s->xres - 1);



    addr = (ram_addr_t) (fb - phys_ram_base);

    start = addr + s->yres * src_width;

    end = addr;

    dirty[0] = dirty[1] = cpu_physical_memory_get_dirty(start, VGA_DIRTY_FLAG);

    for (y = 0; y < s->yres; y ++) {

        new_addr = addr + src_width;

        for (x = addr + TARGET_PAGE_SIZE; x < new_addr;

                        x += TARGET_PAGE_SIZE) {

            dirty[1] = cpu_physical_memory_get_dirty(x, VGA_DIRTY_FLAG);

            dirty[0] |= dirty[1];

        }

        if (dirty[0] || s->invalidated) {

            fn((uint32_t *) s->dma_ch[0].palette,

                            dest, src, s->xres, -dest_width);

            if (addr < start)

                start = addr;

            end = new_addr;

            if (y < *miny)

                *miny = y;

            if (y >= *maxy)

                *maxy = y + 1;

        }

        addr = new_addr;

        dirty[0] = dirty[1];

        src += src_width;

        dest += s->dest_width;

    }



    if (end > start)

        cpu_physical_memory_reset_dirty(start, end, VGA_DIRTY_FLAG);

}
