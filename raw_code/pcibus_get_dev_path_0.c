static char *pcibus_get_dev_path(DeviceState *dev)

{

    PCIDevice *d = container_of(dev, PCIDevice, qdev);

    PCIDevice *t;

    int slot_depth;

    /* Path format: Domain:00:Slot.Function:Slot.Function....:Slot.Function.

     * 00 is added here to make this format compatible with

     * domain:Bus:Slot.Func for systems without nested PCI bridges.

     * Slot.Function list specifies the slot and function numbers for all

     * devices on the path from root to the specific device. */

    int domain_len = strlen("DDDD:00");

    int slot_len = strlen(":SS.F");

    int path_len;

    char *path, *p;



    /* Calculate # of slots on path between device and root. */;

    slot_depth = 0;

    for (t = d; t; t = t->bus->parent_dev) {

        ++slot_depth;

    }



    path_len = domain_len + slot_len * slot_depth;



    /* Allocate memory, fill in the terminating null byte. */

    path = malloc(path_len + 1 /* For '\0' */);

    path[path_len] = '\0';



    /* First field is the domain. */

    snprintf(path, domain_len, "%04x:00", pci_find_domain(d->bus));



    /* Fill in slot numbers. We walk up from device to root, so need to print

     * them in the reverse order, last to first. */

    p = path + path_len;

    for (t = d; t; t = t->bus->parent_dev) {

        p -= slot_len;

        snprintf(p, slot_len, ":%02x.%x", PCI_SLOT(t->devfn), PCI_FUNC(d->devfn));

    }



    return path;

}
