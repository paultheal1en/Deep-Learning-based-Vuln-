static void pm_reset(void *opaque)
{
    ICH9LPCPMRegs *pm = opaque;
    ich9_pm_iospace_update(pm, 0);
    acpi_pm1_evt_reset(&pm->acpi_regs);
    acpi_pm1_cnt_reset(&pm->acpi_regs);
    acpi_pm_tmr_reset(&pm->acpi_regs);
    acpi_gpe_reset(&pm->acpi_regs);
    pm_update_sci(pm);