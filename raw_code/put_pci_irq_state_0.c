static void put_pci_irq_state(QEMUFile *f, void *pv, size_t size)

{

    int i;

    PCIDevice *s = container_of(pv, PCIDevice, config);



    for (i = 0; i < PCI_NUM_PINS; ++i) {

        qemu_put_be32(f, pci_irq_state(s, i));

    }

}
