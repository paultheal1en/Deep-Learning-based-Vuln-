void ich9_pm_init(PCIDevice *lpc_pci, ICH9LPCPMRegs *pm,

                  qemu_irq sci_irq, qemu_irq cmos_s3)

{

    memory_region_init(&pm->io, "ich9-pm", ICH9_PMIO_SIZE);

    memory_region_set_enabled(&pm->io, false);

    memory_region_add_subregion(pci_address_space_io(lpc_pci),

                                0, &pm->io);



    acpi_pm_tmr_init(&pm->acpi_regs, ich9_pm_update_sci_fn, &pm->io);

    acpi_pm1_evt_init(&pm->acpi_regs, ich9_pm_update_sci_fn, &pm->io);

    acpi_pm1_cnt_init(&pm->acpi_regs, &pm->io);



    acpi_gpe_init(&pm->acpi_regs, ICH9_PMIO_GPE0_LEN);

    memory_region_init_io(&pm->io_gpe, &ich9_gpe_ops, pm, "apci-gpe0",

                          ICH9_PMIO_GPE0_LEN);

    memory_region_add_subregion(&pm->io, ICH9_PMIO_GPE0_STS, &pm->io_gpe);



    memory_region_init_io(&pm->io_smi, &ich9_smi_ops, pm, "apci-smi",

                          8);

    memory_region_add_subregion(&pm->io, ICH9_PMIO_SMI_EN, &pm->io_smi);



    pm->irq = sci_irq;

    qemu_register_reset(pm_reset, pm);

    pm->powerdown_notifier.notify = pm_powerdown_req;

    qemu_register_powerdown_notifier(&pm->powerdown_notifier);

}
