static void xilinx_axidma_realize(DeviceState *dev, Error **errp)

{

    XilinxAXIDMA *s = XILINX_AXI_DMA(dev);

    XilinxAXIDMAStreamSlave *ds = XILINX_AXI_DMA_DATA_STREAM(&s->rx_data_dev);

    XilinxAXIDMAStreamSlave *cs = XILINX_AXI_DMA_CONTROL_STREAM(

                                                            &s->rx_control_dev);

    Error *local_err = NULL;



    object_property_add_link(OBJECT(ds), "dma", TYPE_XILINX_AXI_DMA,

                             (Object **)&ds->dma,

                             object_property_allow_set_link,

                             OBJ_PROP_LINK_UNREF_ON_RELEASE,

                             &local_err);

    object_property_add_link(OBJECT(cs), "dma", TYPE_XILINX_AXI_DMA,

                             (Object **)&cs->dma,

                             object_property_allow_set_link,

                             OBJ_PROP_LINK_UNREF_ON_RELEASE,

                             &local_err);

    if (local_err) {

        goto xilinx_axidma_realize_fail;

    }

    object_property_set_link(OBJECT(ds), OBJECT(s), "dma", &local_err);

    object_property_set_link(OBJECT(cs), OBJECT(s), "dma", &local_err);

    if (local_err) {

        goto xilinx_axidma_realize_fail;

    }



    int i;



    for (i = 0; i < 2; i++) {

        struct Stream *st = &s->streams[i];



        st->nr = i;

        st->bh = qemu_bh_new(timer_hit, st);

        st->ptimer = ptimer_init(st->bh, PTIMER_POLICY_DEFAULT);

        ptimer_set_freq(st->ptimer, s->freqhz);

    }

    return;



xilinx_axidma_realize_fail:

    if (!*errp) {

        *errp = local_err;

    }

}
