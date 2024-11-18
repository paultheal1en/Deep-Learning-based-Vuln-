static void enable_interrupt(EEPRO100State * s)

{

    if (!s->int_stat) {

        logout("interrupt enabled\n");

        qemu_irq_raise(s->pci_dev->irq[0]);

        s->int_stat = 1;

    }

}
