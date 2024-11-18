static S390PCIBusDevice *s390_pci_find_dev_by_uid(uint16_t uid)

{

    int i;

    S390PCIBusDevice *pbdev;

    S390pciState *s = s390_get_phb();



    for (i = 0; i < PCI_SLOT_MAX; i++) {

        pbdev = s->pbdev[i];

        if (!pbdev) {

            continue;

        }



        if (pbdev->uid == uid) {

            return pbdev;

        }

    }



    return NULL;

}
