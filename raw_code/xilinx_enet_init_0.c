static void xilinx_enet_init(Object *obj)

{

    XilinxAXIEnet *s = XILINX_AXI_ENET(obj);

    SysBusDevice *sbd = SYS_BUS_DEVICE(obj);

    Error *errp = NULL;



    object_property_add_link(obj, "axistream-connected", TYPE_STREAM_SLAVE,

                             (Object **) &s->tx_dev, &errp);

    assert_no_error(errp);



    object_initialize(&s->rx_data_dev, TYPE_XILINX_AXI_ENET_DATA_STREAM);

    object_property_add_child(OBJECT(s), "axistream-connected-target",

                              (Object *)&s->rx_data_dev, &errp);

    assert_no_error(errp);



    sysbus_init_irq(sbd, &s->irq);



    memory_region_init_io(&s->iomem, &enet_ops, s, "enet", 0x40000);

    sysbus_init_mmio(sbd, &s->iomem);

}
