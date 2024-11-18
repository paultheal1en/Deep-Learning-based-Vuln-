S390PCIBusDevice *s390_pci_find_dev_by_idx(uint32_t idx)

{

    S390PCIBusDevice *pbdev;

    int i;

    int j = 0;

    S390pciState *s = S390_PCI_HOST_BRIDGE(

        object_resolve_path(TYPE_S390_PCI_HOST_BRIDGE, NULL));



    if (!s) {

        return NULL;

    }



    for (i = 0; i < PCI_SLOT_MAX; i++) {

        pbdev = &s->pbdev[i];



        if (pbdev->fh == 0) {

            continue;

        }



        if (j == idx) {

            return pbdev;

        }

        j++;

    }



    return NULL;

}
