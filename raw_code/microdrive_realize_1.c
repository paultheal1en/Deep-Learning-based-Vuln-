static void microdrive_realize(DeviceState *dev, Error **errp)

{

    MicroDriveState *md = MICRODRIVE(dev);



    ide_init2(&md->bus, qemu_allocate_irqs(md_set_irq, md, 1)[0]);

}
