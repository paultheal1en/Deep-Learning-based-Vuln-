static void vt82c686b_init_ports(PCIIDEState *d) {

    int i;

    struct {

        int iobase;

        int iobase2;

        int isairq;

    } port_info[] = {

        {0x1f0, 0x3f6, 14},

        {0x170, 0x376, 15},

    };



    for (i = 0; i < 2; i++) {

        ide_bus_new(&d->bus[i], &d->dev.qdev, i);

        ide_init_ioport(&d->bus[i], port_info[i].iobase, port_info[i].iobase2);

        ide_init2(&d->bus[i], isa_reserve_irq(port_info[i].isairq));



        bmdma_init(&d->bus[i], &d->bmdma[i]);

        d->bmdma[i].bus = &d->bus[i];

        qemu_add_vm_change_state_handler(d->bus[i].dma->ops->restart_cb,

                                         &d->bmdma[i].dma);

    }

}
