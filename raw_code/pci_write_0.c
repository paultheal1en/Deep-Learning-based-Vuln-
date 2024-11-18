static void pci_write(void *opaque, hwaddr addr, uint64_t data,

                      unsigned int size)

{

    AcpiPciHpState *s = opaque;

    switch (addr) {

    case PCI_EJ_BASE:

        if (s->hotplug_select >= ACPI_PCIHP_MAX_HOTPLUG_BUS) {

            break;

        }

        acpi_pcihp_eject_slot(s, s->hotplug_select, data);

        ACPI_PCIHP_DPRINTF("pciej write %" HWADDR_PRIx " <== %" PRIu64 "\n",

                      addr, data);

        break;

    case PCI_SEL_BASE:

        s->hotplug_select = data;

        ACPI_PCIHP_DPRINTF("pcisel write %" HWADDR_PRIx " <== %" PRIu64 "\n",

                      addr, data);

    default:

        break;

    }

}
