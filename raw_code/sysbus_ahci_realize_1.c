static void sysbus_ahci_realize(DeviceState *dev, Error **errp)

{

    SysBusDevice *sbd = SYS_BUS_DEVICE(dev);

    SysbusAHCIState *s = SYSBUS_AHCI(dev);



    ahci_init(&s->ahci, dev, NULL, s->num_ports);



    sysbus_init_mmio(sbd, &s->ahci.mem);

    sysbus_init_irq(sbd, &s->ahci.irq);

}
