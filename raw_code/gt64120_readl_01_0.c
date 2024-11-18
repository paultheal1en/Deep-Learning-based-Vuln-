static uint32_t gt64120_readl (void *opaque,

                               target_phys_addr_t addr)

{

    GT64120State *s = opaque;

    uint32_t val;

    uint32_t saddr;



    val = 0;

    saddr = (addr & 0xfff) >> 2;



    switch (saddr) {



    /* CPU Configuration */

    case GT_MULTI:

        /* Only one GT64xxx is present on the CPU bus, return

           the initial value */

        val = s->regs[saddr];

        break;



    /* CPU Error Report */

    case GT_CPUERR_ADDRLO:

    case GT_CPUERR_ADDRHI:

    case GT_CPUERR_DATALO:

    case GT_CPUERR_DATAHI:

    case GT_CPUERR_PARITY:

        /* Emulated memory has no error, always return the initial

           values */

        val = s->regs[saddr];

        break;



    /* CPU Sync Barrier */

    case GT_PCI0SYNC:

    case GT_PCI1SYNC:

        /* Reading those register should empty all FIFO on the PCI

           bus, which are not emulated. The return value should be

           a random value that should be ignored. */

        val = 0xc000ffee;

        break;



    /* ECC */

    case GT_ECC_ERRDATALO:

    case GT_ECC_ERRDATAHI:

    case GT_ECC_MEM:

    case GT_ECC_CALC:

    case GT_ECC_ERRADDR:

        /* Emulated memory has no error, always return the initial

           values */

        val = s->regs[saddr];

        break;



    case GT_CPU:

    case GT_SCS10LD:

    case GT_SCS10HD:

    case GT_SCS32LD:

    case GT_SCS32HD:

    case GT_CS20LD:

    case GT_CS20HD:

    case GT_CS3BOOTLD:

    case GT_CS3BOOTHD:

    case GT_SCS10AR:

    case GT_SCS32AR:

    case GT_CS20R:

    case GT_CS3BOOTR:

    case GT_PCI0IOLD:

    case GT_PCI0M0LD:

    case GT_PCI0M1LD:

    case GT_PCI1IOLD:

    case GT_PCI1M0LD:

    case GT_PCI1M1LD:

    case GT_PCI0IOHD:

    case GT_PCI0M0HD:

    case GT_PCI0M1HD:

    case GT_PCI1IOHD:

    case GT_PCI1M0HD:

    case GT_PCI1M1HD:

    case GT_PCI0IOREMAP:

    case GT_PCI0M0REMAP:

    case GT_PCI0M1REMAP:

    case GT_PCI1IOREMAP:

    case GT_PCI1M0REMAP:

    case GT_PCI1M1REMAP:

    case GT_ISD:

        val = s->regs[saddr];

        break;

    case GT_PCI0_IACK:

        /* Read the IRQ number */

        val = pic_read_irq(isa_pic);

        break;



    /* SDRAM and Device Address Decode */

    case GT_SCS0LD:

    case GT_SCS0HD:

    case GT_SCS1LD:

    case GT_SCS1HD:

    case GT_SCS2LD:

    case GT_SCS2HD:

    case GT_SCS3LD:

    case GT_SCS3HD:

    case GT_CS0LD:

    case GT_CS0HD:

    case GT_CS1LD:

    case GT_CS1HD:

    case GT_CS2LD:

    case GT_CS2HD:

    case GT_CS3LD:

    case GT_CS3HD:

    case GT_BOOTLD:

    case GT_BOOTHD:

    case GT_ADERR:

        val = s->regs[saddr];

        break;



    /* SDRAM Configuration */

    case GT_SDRAM_CFG:

    case GT_SDRAM_OPMODE:

    case GT_SDRAM_BM:

    case GT_SDRAM_ADDRDECODE:

        val = s->regs[saddr];

        break;



    /* SDRAM Parameters */

    case GT_SDRAM_B0:

    case GT_SDRAM_B1:

    case GT_SDRAM_B2:

    case GT_SDRAM_B3:

        /* We don't simulate electrical parameters of the SDRAM.

           Just return the last written value. */

        val = s->regs[saddr];

        break;



    /* Device Parameters */

    case GT_DEV_B0:

    case GT_DEV_B1:

    case GT_DEV_B2:

    case GT_DEV_B3:

    case GT_DEV_BOOT:

        val = s->regs[saddr];

        break;



    /* DMA Record */

    case GT_DMA0_CNT:

    case GT_DMA1_CNT:

    case GT_DMA2_CNT:

    case GT_DMA3_CNT:

    case GT_DMA0_SA:

    case GT_DMA1_SA:

    case GT_DMA2_SA:

    case GT_DMA3_SA:

    case GT_DMA0_DA:

    case GT_DMA1_DA:

    case GT_DMA2_DA:

    case GT_DMA3_DA:

    case GT_DMA0_NEXT:

    case GT_DMA1_NEXT:

    case GT_DMA2_NEXT:

    case GT_DMA3_NEXT:

    case GT_DMA0_CUR:

    case GT_DMA1_CUR:

    case GT_DMA2_CUR:

    case GT_DMA3_CUR:

        val = s->regs[saddr];

        break;



    /* DMA Channel Control */

    case GT_DMA0_CTRL:

    case GT_DMA1_CTRL:

    case GT_DMA2_CTRL:

    case GT_DMA3_CTRL:

        val = s->regs[saddr];

        break;



    /* DMA Arbiter */

    case GT_DMA_ARB:

        val = s->regs[saddr];

        break;



    /* Timer/Counter */

    case GT_TC0:

    case GT_TC1:

    case GT_TC2:

    case GT_TC3:

    case GT_TC_CONTROL:

        val = s->regs[saddr];

        break;



    /* PCI Internal */

    case GT_PCI0_CFGADDR:

        val = s->pci->config_reg;

        break;

    case GT_PCI0_CFGDATA:

        if (!(s->pci->config_reg & (1u << 31)))

            val = 0xffffffff;

        else

            val = pci_host_data_readl(s->pci, 0);

        break;



    case GT_PCI0_CMD:

    case GT_PCI0_TOR:

    case GT_PCI0_BS_SCS10:

    case GT_PCI0_BS_SCS32:

    case GT_PCI0_BS_CS20:

    case GT_PCI0_BS_CS3BT:

    case GT_PCI1_IACK:

    case GT_PCI0_BARE:

    case GT_PCI0_PREFMBR:

    case GT_PCI0_SCS10_BAR:

    case GT_PCI0_SCS32_BAR:

    case GT_PCI0_CS20_BAR:

    case GT_PCI0_CS3BT_BAR:

    case GT_PCI0_SSCS10_BAR:

    case GT_PCI0_SSCS32_BAR:

    case GT_PCI0_SCS3BT_BAR:

    case GT_PCI1_CMD:

    case GT_PCI1_TOR:

    case GT_PCI1_BS_SCS10:

    case GT_PCI1_BS_SCS32:

    case GT_PCI1_BS_CS20:

    case GT_PCI1_BS_CS3BT:

    case GT_PCI1_BARE:

    case GT_PCI1_PREFMBR:

    case GT_PCI1_SCS10_BAR:

    case GT_PCI1_SCS32_BAR:

    case GT_PCI1_CS20_BAR:

    case GT_PCI1_CS3BT_BAR:

    case GT_PCI1_SSCS10_BAR:

    case GT_PCI1_SSCS32_BAR:

    case GT_PCI1_SCS3BT_BAR:

    case GT_PCI1_CFGADDR:

    case GT_PCI1_CFGDATA:

        val = s->regs[saddr];

        break;



    /* Interrupts */

    case GT_INTRCAUSE:

        val = s->regs[saddr];

        dprintf("INTRCAUSE %x\n", val);

        break;

    case GT_INTRMASK:

        val = s->regs[saddr];

        dprintf("INTRMASK %x\n", val);

        break;

    case GT_PCI0_ICMASK:

        val = s->regs[saddr];

        dprintf("ICMASK %x\n", val);

        break;

    case GT_PCI0_SERR0MASK:

        val = s->regs[saddr];

        dprintf("SERR0MASK %x\n", val);

        break;



    /* Reserved when only PCI_0 is configured. */

    case GT_HINTRCAUSE:

    case GT_CPU_INTSEL:

    case GT_PCI0_INTSEL:

    case GT_HINTRMASK:

    case GT_PCI0_HICMASK:

    case GT_PCI1_SERR1MASK:

        val = s->regs[saddr];

        break;



    default:

        val = s->regs[saddr];

        dprintf ("Bad register offset 0x%x\n", (int)addr);

        break;

    }



#ifdef TARGET_WORDS_BIGENDIAN

    val = bswap32(val);

#endif

    return val;

}
