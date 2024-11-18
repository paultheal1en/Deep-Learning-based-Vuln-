static void megasas_scsi_uninit(PCIDevice *d)

{

    MegasasState *s = MEGASAS(d);



    if (megasas_use_msix(s)) {

        msix_uninit(d, &s->mmio_io, &s->mmio_io);

    }

    if (megasas_use_msi(s)) {

        msi_uninit(d);

    }

}
