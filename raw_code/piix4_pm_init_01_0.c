I2CBus *piix4_pm_init(PCIBus *bus, int devfn, uint32_t smb_io_base,

                      qemu_irq sci_irq, qemu_irq smi_irq,

                      int kvm_enabled, FWCfgState *fw_cfg,

                      DeviceState **piix4_pm)

{

    DeviceState *dev;

    PIIX4PMState *s;



    dev = DEVICE(pci_create(bus, devfn, TYPE_PIIX4_PM));

    qdev_prop_set_uint32(dev, "smb_io_base", smb_io_base);

    if (piix4_pm) {

        *piix4_pm = dev;

    }



    s = PIIX4_PM(dev);

    s->irq = sci_irq;

    s->smi_irq = smi_irq;

    s->kvm_enabled = kvm_enabled;

    if (xen_enabled()) {

        s->use_acpi_pci_hotplug = false;

    }



    qdev_init_nofail(dev);



    if (fw_cfg) {

        uint8_t suspend[6] = {128, 0, 0, 129, 128, 128};

        suspend[3] = 1 | ((!s->disable_s3) << 7);

        suspend[4] = s->s4_val | ((!s->disable_s4) << 7);



        fw_cfg_add_file(fw_cfg, "etc/system-states", g_memdup(suspend, 6), 6);

    }



    return s->smb.smbus;

}
