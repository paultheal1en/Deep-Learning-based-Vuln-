static void pci_bridge_write_config(PCIDevice *d,

                             uint32_t address, uint32_t val, int len)

{

    PCIBridge *s = (PCIBridge *)d;



    if (address == 0x19 || (address == 0x18 && len > 1)) {

        if (address == 0x19)

            s->bus->bus_num = val & 0xff;

        else

            s->bus->bus_num = (val >> 8) & 0xff;

#if defined(DEBUG_PCI)

        printf ("pci-bridge: %s: Assigned bus %d\n", d->name, s->bus->bus_num);

#endif

    }

    pci_default_write_config(d, address, val, len);

}
