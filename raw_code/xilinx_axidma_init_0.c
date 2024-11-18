static void xilinx_axidma_init(Object *obj)

{

    XilinxAXIDMA *s = XILINX_AXI_DMA(obj);

    SysBusDevice *sbd = SYS_BUS_DEVICE(obj);

    Error *errp = NULL;



    object_property_add_link(obj, "axistream-connected", TYPE_STREAM_SLAVE,

                             (Object **) &s->tx_dev, NULL);



    object_initialize(&s->rx_data_dev, TYPE_XILINX_AXI_DMA_DATA_STREAM);

    object_property_add_child(OBJECT(s), "axistream-connected-target",

                              (Object *)&s->rx_data_dev, &errp);

    assert_no_error(errp);



    sysbus_init_irq(sbd, &s->streams[0].irq);

    sysbus_init_irq(sbd, &s->streams[1].irq);



    memory_region_init_io(&s->iomem, &axidma_ops, s,

                          "xlnx.axi-dma", R_MAX * 4 * 2);

    sysbus_init_mmio(sbd, &s->iomem);

}
