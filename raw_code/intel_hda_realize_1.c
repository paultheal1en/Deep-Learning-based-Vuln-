static void intel_hda_realize(PCIDevice *pci, Error **errp)

{

    IntelHDAState *d = INTEL_HDA(pci);

    uint8_t *conf = d->pci.config;



    d->name = object_get_typename(OBJECT(d));



    pci_config_set_interrupt_pin(conf, 1);



    /* HDCTL off 0x40 bit 0 selects signaling mode (1-HDA, 0 - Ac97) 18.1.19 */

    conf[0x40] = 0x01;



    memory_region_init_io(&d->mmio, OBJECT(d), &intel_hda_mmio_ops, d,

                          "intel-hda", 0x4000);

    pci_register_bar(&d->pci, 0, 0, &d->mmio);

    if (d->msi != ON_OFF_AUTO_OFF) {

         /* TODO check for errors */

        msi_init(&d->pci, d->old_msi_addr ? 0x50 : 0x60, 1, true, false);

    }



    hda_codec_bus_init(DEVICE(pci), &d->codecs, sizeof(d->codecs),

                       intel_hda_response, intel_hda_xfer);

}
