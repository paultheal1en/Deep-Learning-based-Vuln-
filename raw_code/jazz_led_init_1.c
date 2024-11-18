static int jazz_led_init(SysBusDevice *dev)

{

    LedState *s = FROM_SYSBUS(LedState, dev);



    memory_region_init_io(&s->iomem, &led_ops, s, "led", 1);

    sysbus_init_mmio(dev, &s->iomem);



    s->ds = graphic_console_init(jazz_led_update_display,

                                 jazz_led_invalidate_display,

                                 jazz_led_screen_dump,

                                 jazz_led_text_update, s);



    return 0;

}
