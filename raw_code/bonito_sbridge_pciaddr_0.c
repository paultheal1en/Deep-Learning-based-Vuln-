static uint32_t bonito_sbridge_pciaddr(void *opaque, target_phys_addr_t addr)

{

    PCIBonitoState *s = opaque;

    PCIHostState *phb = PCI_HOST_BRIDGE(s->pcihost);

    uint32_t cfgaddr;

    uint32_t idsel;

    uint32_t devno;

    uint32_t funno;

    uint32_t regno;

    uint32_t pciaddr;



    /* support type0 pci config */

    if ((s->regs[BONITO_PCIMAP_CFG] & 0x10000) != 0x0) {

        return 0xffffffff;

    }



    cfgaddr = addr & 0xffff;

    cfgaddr |= (s->regs[BONITO_PCIMAP_CFG] & 0xffff) << 16;



    idsel = (cfgaddr & BONITO_PCICONF_IDSEL_MASK) >> BONITO_PCICONF_IDSEL_OFFSET;

    devno = ffs(idsel) - 1;

    funno = (cfgaddr & BONITO_PCICONF_FUN_MASK) >> BONITO_PCICONF_FUN_OFFSET;

    regno = (cfgaddr & BONITO_PCICONF_REG_MASK) >> BONITO_PCICONF_REG_OFFSET;



    if (idsel == 0) {

        fprintf(stderr, "error in bonito pci config address " TARGET_FMT_plx

            ",pcimap_cfg=%x\n", addr, s->regs[BONITO_PCIMAP_CFG]);

        exit(1);

    }

    pciaddr = PCI_ADDR(pci_bus_num(phb->bus), devno, funno, regno);

    DPRINTF("cfgaddr %x pciaddr %x busno %x devno %d funno %d regno %d\n",

        cfgaddr, pciaddr, pci_bus_num(phb->bus), devno, funno, regno);



    return pciaddr;

}
