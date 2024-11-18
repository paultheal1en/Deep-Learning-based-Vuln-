void acpi_pm1_cnt_init(ACPIREGS *ar, MemoryRegion *parent)

{

    ar->wakeup.notify = acpi_notify_wakeup;

    qemu_register_wakeup_notifier(&ar->wakeup);

    memory_region_init_io(&ar->pm1.cnt.io, &acpi_pm_cnt_ops, ar, "acpi-cnt", 2);

    memory_region_add_subregion(parent, 4, &ar->pm1.cnt.io);

}
