static void s390_pcihost_init_as(S390pciState *s)

{

    int i;

    S390PCIBusDevice *pbdev;



    for (i = 0; i < PCI_SLOT_MAX; i++) {

        pbdev = &s->pbdev[i];

        memory_region_init(&pbdev->mr, OBJECT(s),

                           "iommu-root-s390", UINT64_MAX);

        address_space_init(&pbdev->as, &pbdev->mr, "iommu-pci");

    }



    memory_region_init_io(&s->msix_notify_mr, OBJECT(s),

                          &s390_msi_ctrl_ops, s, "msix-s390", UINT64_MAX);

    address_space_init(&s->msix_notify_as, &s->msix_notify_mr, "msix-pci");

}
