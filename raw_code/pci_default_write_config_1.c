void pci_default_write_config(PCIDevice *d, uint32_t addr, uint32_t val, int l)

{

    int i, was_irq_disabled = pci_irq_disabled(d);

    uint32_t config_size = pci_config_size(d);



    for (i = 0; i < l && addr + i < config_size; val >>= 8, ++i) {

        uint8_t wmask = d->wmask[addr + i];

        uint8_t w1cmask = d->w1cmask[addr + i];

        assert(!(wmask & w1cmask));

        d->config[addr + i] = (d->config[addr + i] & ~wmask) | (val & wmask);

        d->config[addr + i] &= ~(val & w1cmask); /* W1C: Write 1 to Clear */

    }

    if (ranges_overlap(addr, l, PCI_BASE_ADDRESS_0, 24) ||

        ranges_overlap(addr, l, PCI_ROM_ADDRESS, 4) ||

        ranges_overlap(addr, l, PCI_ROM_ADDRESS1, 4) ||

        range_covers_byte(addr, l, PCI_COMMAND))

        pci_update_mappings(d);



    if (range_covers_byte(addr, l, PCI_COMMAND))

        pci_update_irq_disabled(d, was_irq_disabled);

}
