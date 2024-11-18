static uint32_t bonito_spciconf_readl(void *opaque, target_phys_addr_t addr)

{

    PCIBonitoState *s = opaque;

    PCIDevice *d = PCI_DEVICE(s);

    PCIHostState *phb = PCI_HOST_BRIDGE(s->pcihost);

    uint32_t pciaddr;

    uint16_t status;



    DPRINTF("bonito_spciconf_readl "TARGET_FMT_plx"\n", addr);

    assert((addr & 0x3) == 0);



    pciaddr = bonito_sbridge_pciaddr(s, addr);



    if (pciaddr == 0xffffffff) {

        return 0xffffffff;

    }



    /* set the pci address in s->config_reg */

    phb->config_reg = (pciaddr) | (1u << 31);



    /* clear PCI_STATUS_REC_MASTER_ABORT and PCI_STATUS_REC_TARGET_ABORT */

    status = pci_get_word(d->config + PCI_STATUS);

    status &= ~(PCI_STATUS_REC_MASTER_ABORT | PCI_STATUS_REC_TARGET_ABORT);

    pci_set_word(d->config + PCI_STATUS, status);



    return pci_data_read(phb->bus, phb->config_reg, 4);

}
