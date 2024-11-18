S390PCIBusDevice *s390_pci_find_dev_by_fh(uint32_t fh)

{

    S390PCIBusDevice *pbdev;

    int i;

    S390pciState *s = S390_PCI_HOST_BRIDGE(

        object_resolve_path(TYPE_S390_PCI_HOST_BRIDGE, NULL));



    if (!s || !fh) {

        return NULL;

    }



    for (i = 0; i < PCI_SLOT_MAX; i++) {

        pbdev = &s->pbdev[i];

        if (pbdev->fh == fh) {

            return pbdev;

        }

    }



    return NULL;

}
