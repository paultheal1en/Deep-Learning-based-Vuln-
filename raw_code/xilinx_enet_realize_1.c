static void xilinx_enet_realize(DeviceState *dev, Error **errp)

{

    XilinxAXIEnet *s = XILINX_AXI_ENET(dev);

    XilinxAXIEnetStreamSlave *ds = XILINX_AXI_ENET_DATA_STREAM(&s->rx_data_dev);

    XilinxAXIEnetStreamSlave *cs = XILINX_AXI_ENET_CONTROL_STREAM(

                                                            &s->rx_control_dev);

    Error *local_errp = NULL;



    object_property_add_link(OBJECT(ds), "enet", "xlnx.axi-ethernet",

                             (Object **) &ds->enet,


                             OBJ_PROP_LINK_UNREF_ON_RELEASE,

                             &local_errp);

    object_property_add_link(OBJECT(cs), "enet", "xlnx.axi-ethernet",

                             (Object **) &cs->enet,


                             OBJ_PROP_LINK_UNREF_ON_RELEASE,

                             &local_errp);

    if (local_errp) {

        goto xilinx_enet_realize_fail;

    }

    object_property_set_link(OBJECT(ds), OBJECT(s), "enet", &local_errp);

    object_property_set_link(OBJECT(cs), OBJECT(s), "enet", &local_errp);

    if (local_errp) {

        goto xilinx_enet_realize_fail;

    }



    qemu_macaddr_default_if_unset(&s->conf.macaddr);

    s->nic = qemu_new_nic(&net_xilinx_enet_info, &s->conf,

                          object_get_typename(OBJECT(dev)), dev->id, s);

    qemu_format_nic_info_str(qemu_get_queue(s->nic), s->conf.macaddr.a);



    tdk_init(&s->TEMAC.phy);

    mdio_attach(&s->TEMAC.mdio_bus, &s->TEMAC.phy, s->c_phyaddr);



    s->TEMAC.parent = s;



    s->rxmem = g_malloc(s->c_rxmem);

    return;



xilinx_enet_realize_fail:

    if (!*errp) {

        *errp = local_errp;

    }

}