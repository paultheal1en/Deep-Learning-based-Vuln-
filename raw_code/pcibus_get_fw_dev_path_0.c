static char *pcibus_get_fw_dev_path(DeviceState *dev)

{

    PCIDevice *d = (PCIDevice *)dev;

    char path[50], name[33];

    int off;



    off = snprintf(path, sizeof(path), "%s@%x",

                   pci_dev_fw_name(dev, name, sizeof name),

                   PCI_SLOT(d->devfn));

    if (PCI_FUNC(d->devfn))

        snprintf(path + off, sizeof(path) + off, ",%x", PCI_FUNC(d->devfn));

    return strdup(path);

}
