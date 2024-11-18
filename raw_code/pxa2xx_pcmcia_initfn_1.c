static void pxa2xx_pcmcia_initfn(Object *obj)

{

    SysBusDevice *sbd = SYS_BUS_DEVICE(obj);

    PXA2xxPCMCIAState *s = PXA2XX_PCMCIA(obj);



    memory_region_init(&s->container_mem, obj, "container", 0x10000000);

    sysbus_init_mmio(sbd, &s->container_mem);



    /* Socket I/O Memory Space */

    memory_region_init_io(&s->iomem, NULL, &pxa2xx_pcmcia_io_ops, s,

                          "pxa2xx-pcmcia-io", 0x04000000);

    memory_region_add_subregion(&s->container_mem, 0x00000000,

                                &s->iomem);



    /* Then next 64 MB is reserved */



    /* Socket Attribute Memory Space */

    memory_region_init_io(&s->attr_iomem, NULL, &pxa2xx_pcmcia_attr_ops, s,

                          "pxa2xx-pcmcia-attribute", 0x04000000);

    memory_region_add_subregion(&s->container_mem, 0x08000000,

                                &s->attr_iomem);



    /* Socket Common Memory Space */

    memory_region_init_io(&s->common_iomem, NULL, &pxa2xx_pcmcia_common_ops, s,

                          "pxa2xx-pcmcia-common", 0x04000000);

    memory_region_add_subregion(&s->container_mem, 0x0c000000,

                                &s->common_iomem);



    s->slot.irq = qemu_allocate_irqs(pxa2xx_pcmcia_set_irq, s, 1)[0];



    object_property_add_link(obj, "card", TYPE_PCMCIA_CARD,

                             (Object **)&s->card,

                             NULL, /* read-only property */

                             0, NULL);

}
