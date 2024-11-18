vmxnet3_init_msix(VMXNET3State *s)

{

    PCIDevice *d = PCI_DEVICE(s);

    int res = msix_init(d, VMXNET3_MAX_INTRS,

                        &s->msix_bar,

                        VMXNET3_MSIX_BAR_IDX, VMXNET3_OFF_MSIX_TABLE,

                        &s->msix_bar,

                        VMXNET3_MSIX_BAR_IDX, VMXNET3_OFF_MSIX_PBA(s),

                        VMXNET3_MSIX_OFFSET(s));



    if (0 > res) {

        VMW_WRPRN("Failed to initialize MSI-X, error %d", res);

        s->msix_used = false;

    } else {

        if (!vmxnet3_use_msix_vectors(s, VMXNET3_MAX_INTRS)) {

            VMW_WRPRN("Failed to use MSI-X vectors, error %d", res);

            msix_uninit(d, &s->msix_bar, &s->msix_bar);

            s->msix_used = false;

        } else {

            s->msix_used = true;

        }

    }

    return s->msix_used;

}
