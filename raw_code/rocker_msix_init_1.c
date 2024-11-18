static int rocker_msix_init(Rocker *r)

{

    PCIDevice *dev = PCI_DEVICE(r);

    int err;



    err = msix_init(dev, ROCKER_MSIX_VEC_COUNT(r->fp_ports),

                    &r->msix_bar,

                    ROCKER_PCI_MSIX_BAR_IDX, ROCKER_PCI_MSIX_TABLE_OFFSET,

                    &r->msix_bar,

                    ROCKER_PCI_MSIX_BAR_IDX, ROCKER_PCI_MSIX_PBA_OFFSET,

                    0);

    if (err) {

        return err;

    }



    err = rocker_msix_vectors_use(r, ROCKER_MSIX_VEC_COUNT(r->fp_ports));

    if (err) {

        goto err_msix_vectors_use;

    }



    return 0;



err_msix_vectors_use:

    msix_uninit(dev, &r->msix_bar, &r->msix_bar);

    return err;

}
