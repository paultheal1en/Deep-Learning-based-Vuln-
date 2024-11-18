static void multi_serial_pci_realize(PCIDevice *dev, Error **errp)

{

    PCIDeviceClass *pc = PCI_DEVICE_GET_CLASS(dev);

    PCIMultiSerialState *pci = DO_UPCAST(PCIMultiSerialState, dev, dev);

    SerialState *s;

    Error *err = NULL;

    int i;



    switch (pc->device_id) {

    case 0x0003:

        pci->ports = 2;

        break;

    case 0x0004:

        pci->ports = 4;

        break;

    }

    assert(pci->ports > 0);

    assert(pci->ports <= PCI_SERIAL_MAX_PORTS);



    pci->dev.config[PCI_CLASS_PROG] = pci->prog_if;

    pci->dev.config[PCI_INTERRUPT_PIN] = 0x01;

    memory_region_init(&pci->iobar, OBJECT(pci), "multiserial", 8 * pci->ports);

    pci_register_bar(&pci->dev, 0, PCI_BASE_ADDRESS_SPACE_IO, &pci->iobar);

    pci->irqs = qemu_allocate_irqs(multi_serial_irq_mux, pci,

                                   pci->ports);



    for (i = 0; i < pci->ports; i++) {

        s = pci->state + i;

        s->baudbase = 115200;

        serial_realize_core(s, &err);

        if (err != NULL) {

            error_propagate(errp, err);

            return;

        }

        s->irq = pci->irqs[i];

        pci->name[i] = g_strdup_printf("uart #%d", i+1);

        memory_region_init_io(&s->io, OBJECT(pci), &serial_io_ops, s,

                              pci->name[i], 8);

        memory_region_add_subregion(&pci->iobar, 8 * i, &s->io);

    }

}
