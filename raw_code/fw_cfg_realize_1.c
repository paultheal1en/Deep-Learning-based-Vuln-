static void fw_cfg_realize(DeviceState *dev, Error **errp)

{

    FWCfgState *s = FW_CFG(dev);

    SysBusDevice *sbd = SYS_BUS_DEVICE(dev);



    if (s->ctl_iobase + 1 == s->data_iobase) {

        sysbus_add_io(sbd, s->ctl_iobase, &s->comb_iomem);

    } else {

        if (s->ctl_iobase) {

            sysbus_add_io(sbd, s->ctl_iobase, &s->ctl_iomem);

        }

        if (s->data_iobase) {

            sysbus_add_io(sbd, s->data_iobase, &s->data_iomem);

        }

    }

}
