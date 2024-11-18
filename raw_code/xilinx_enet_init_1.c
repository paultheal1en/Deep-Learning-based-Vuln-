static void xilinx_enet_init(Object *obj)

{

    XilinxAXIEnet *s = XILINX_AXI_ENET(obj);

    SysBusDevice *sbd = SYS_BUS_DEVICE(obj);



    object_property_add_link(obj, "axistream-connected", TYPE_STREAM_SLAVE,

                             (Object **) &s->tx_data_dev, &error_abort);

    object_property_add_link(obj, "axistream-control-connected",

                             TYPE_STREAM_SLAVE,

                             (Object **) &s->tx_control_dev, &error_abort);



    object_initialize(&s->rx_data_dev, sizeof(s->rx_data_dev),

                      TYPE_XILINX_AXI_ENET_DATA_STREAM);

    object_initialize(&s->rx_control_dev, sizeof(s->rx_control_dev),

                      TYPE_XILINX_AXI_ENET_CONTROL_STREAM);

    object_property_add_child(OBJECT(s), "axistream-connected-target",

                              (Object *)&s->rx_data_dev, &error_abort);

    object_property_add_child(OBJECT(s), "axistream-control-connected-target",

                              (Object *)&s->rx_control_dev, &error_abort);



    sysbus_init_irq(sbd, &s->irq);



    memory_region_init_io(&s->iomem, OBJECT(s), &enet_ops, s, "enet", 0x40000);

    sysbus_init_mmio(sbd, &s->iomem);

}
