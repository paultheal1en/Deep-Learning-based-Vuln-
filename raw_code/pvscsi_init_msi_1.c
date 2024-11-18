pvscsi_init_msi(PVSCSIState *s)

{

    int res;

    PCIDevice *d = PCI_DEVICE(s);



    res = msi_init(d, PVSCSI_MSI_OFFSET(s), PVSCSI_MSIX_NUM_VECTORS,

                   PVSCSI_USE_64BIT, PVSCSI_PER_VECTOR_MASK);

    if (res < 0) {

        trace_pvscsi_init_msi_fail(res);

        s->msi_used = false;

    } else {

        s->msi_used = true;

    }

}
