static void enable_device(AcpiPciHpState *s, unsigned bsel, int slot)

{

    s->acpi_pcihp_pci_status[bsel].device_present |= (1U << slot);

}
