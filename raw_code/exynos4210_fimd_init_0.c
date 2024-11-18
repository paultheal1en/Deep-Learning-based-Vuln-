static int exynos4210_fimd_init(SysBusDevice *dev)

{

    Exynos4210fimdState *s = FROM_SYSBUS(Exynos4210fimdState, dev);



    s->ifb = NULL;



    sysbus_init_irq(dev, &s->irq[0]);

    sysbus_init_irq(dev, &s->irq[1]);

    sysbus_init_irq(dev, &s->irq[2]);



    memory_region_init_io(&s->iomem, &exynos4210_fimd_mmio_ops, s,

            "exynos4210.fimd", FIMD_REGS_SIZE);

    sysbus_init_mmio(dev, &s->iomem);

    s->console = graphic_console_init(exynos4210_fimd_update,

                                  exynos4210_fimd_invalidate, NULL, NULL, s);



    return 0;

}
