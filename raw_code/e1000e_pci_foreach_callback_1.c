static void e1000e_pci_foreach_callback(QPCIDevice *dev, int devfn, void *data)

{

    *(QPCIDevice **) data = dev;

}
