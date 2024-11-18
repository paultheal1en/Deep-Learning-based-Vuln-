void gt64120_reset(void *opaque)

{

    GT64120State *s = opaque;



    /* CPU Configuration */

#ifdef TARGET_WORDS_BIGENDIAN

    s->regs[GT_CPU]           = 0x00000000;

#else

    s->regs[GT_CPU]           = 0x00001000;

#endif

    s->regs[GT_MULTI]         = 0x00000000;



    /* CPU Address decode FIXME: not complete*/

    s->regs[GT_PCI0IOLD]      = 0x00000080;

    s->regs[GT_PCI0IOHD]      = 0x0000000f;

    s->regs[GT_PCI0M0LD]      = 0x00000090;

    s->regs[GT_PCI0M0HD]      = 0x0000001f;

    s->regs[GT_PCI0M1LD]      = 0x00000790;

    s->regs[GT_PCI0M1HD]      = 0x0000001f;

    s->regs[GT_PCI1IOLD]      = 0x00000100;

    s->regs[GT_PCI1IOHD]      = 0x0000000f;

    s->regs[GT_PCI1M0LD]      = 0x00000110;

    s->regs[GT_PCI1M0HD]      = 0x0000001f;

    s->regs[GT_PCI1M1LD]      = 0x00000120;

    s->regs[GT_PCI1M1HD]      = 0x0000002f;

    s->regs[GT_PCI0IOREMAP]   = 0x00000080;

    s->regs[GT_PCI0M0REMAP]   = 0x00000090;

    s->regs[GT_PCI0M1REMAP]   = 0x00000790;

    s->regs[GT_PCI1IOREMAP]   = 0x00000100;

    s->regs[GT_PCI1M0REMAP]   = 0x00000110;

    s->regs[GT_PCI1M1REMAP]   = 0x00000120;



    /* CPU Error Report */

    s->regs[GT_CPUERR_ADDRLO] = 0x00000000;

    s->regs[GT_CPUERR_ADDRHI] = 0x00000000;

    s->regs[GT_CPUERR_DATALO] = 0xffffffff;

    s->regs[GT_CPUERR_DATAHI] = 0xffffffff;

    s->regs[GT_CPUERR_PARITY] = 0x000000ff;



    /* ECC */

    s->regs[GT_ECC_ERRDATALO] = 0x00000000;

    s->regs[GT_ECC_ERRDATAHI] = 0x00000000;

    s->regs[GT_ECC_MEM]       = 0x00000000;

    s->regs[GT_ECC_CALC]      = 0x00000000;

    s->regs[GT_ECC_ERRADDR]   = 0x00000000;



    /* SDRAM Parameters */

    s->regs[GT_SDRAM_B0]      = 0x00000005;    

    s->regs[GT_SDRAM_B1]      = 0x00000005;    

    s->regs[GT_SDRAM_B2]      = 0x00000005;    

    s->regs[GT_SDRAM_B3]      = 0x00000005;    



    /* PCI Internal FIXME: not complete*/

#ifdef TARGET_WORDS_BIGENDIAN

    s->regs[GT_PCI0_CMD]      = 0x00000000;

    s->regs[GT_PCI1_CMD]      = 0x00000000;

#else

    s->regs[GT_PCI0_CMD]      = 0x00010001;

    s->regs[GT_PCI1_CMD]      = 0x00010001;

#endif

    s->regs[GT_PCI0_IACK]     = 0x00000000;

    s->regs[GT_PCI1_IACK]     = 0x00000000;



    gt64120_pci_mapping(s);

}
