void s390_pci_iommu_disable(S390PCIBusDevice *pbdev)

{

    memory_region_del_subregion(&pbdev->mr, &pbdev->iommu_mr);

    object_unparent(OBJECT(&pbdev->iommu_mr));

    pbdev->iommu_enabled = false;

}
