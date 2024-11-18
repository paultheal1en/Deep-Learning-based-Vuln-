PXA2xxLCDState *pxa2xx_lcdc_init(MemoryRegion *sysmem,

                                 target_phys_addr_t base, qemu_irq irq)

{

    PXA2xxLCDState *s;



    s = (PXA2xxLCDState *) g_malloc0(sizeof(PXA2xxLCDState));

    s->invalidated = 1;

    s->irq = irq;

    s->sysmem = sysmem;



    pxa2xx_lcdc_orientation(s, graphic_rotate);



    memory_region_init_io(&s->iomem, &pxa2xx_lcdc_ops, s,

                          "pxa2xx-lcd-controller", 0x00100000);

    memory_region_add_subregion(sysmem, base, &s->iomem);



    s->ds = graphic_console_init(pxa2xx_update_display,

                                 pxa2xx_invalidate_display,

                                 pxa2xx_screen_dump, NULL, s);



    switch (ds_get_bits_per_pixel(s->ds)) {

    case 0:

        s->dest_width = 0;

        break;

    case 8:

        s->line_fn[0] = pxa2xx_draw_fn_8;

        s->line_fn[1] = pxa2xx_draw_fn_8t;

        s->dest_width = 1;

        break;

    case 15:

        s->line_fn[0] = pxa2xx_draw_fn_15;

        s->line_fn[1] = pxa2xx_draw_fn_15t;

        s->dest_width = 2;

        break;

    case 16:

        s->line_fn[0] = pxa2xx_draw_fn_16;

        s->line_fn[1] = pxa2xx_draw_fn_16t;

        s->dest_width = 2;

        break;

    case 24:

        s->line_fn[0] = pxa2xx_draw_fn_24;

        s->line_fn[1] = pxa2xx_draw_fn_24t;

        s->dest_width = 3;

        break;

    case 32:

        s->line_fn[0] = pxa2xx_draw_fn_32;

        s->line_fn[1] = pxa2xx_draw_fn_32t;

        s->dest_width = 4;

        break;

    default:

        fprintf(stderr, "%s: Bad color depth\n", __FUNCTION__);

        exit(1);

    }



    vmstate_register(NULL, 0, &vmstate_pxa2xx_lcdc, s);



    return s;

}
