static void gt64120_pci_mapping(GT64120State *s)

{

    /* Update IO mapping */

    if ((s->regs[GT_PCI0IOLD] & 0x7f) <= s->regs[GT_PCI0IOHD])

    {

      /* Unmap old IO address */

      if (s->PCI0IO_length)

      {

          memory_region_del_subregion(get_system_memory(), &s->PCI0IO_mem);

          memory_region_destroy(&s->PCI0IO_mem);

      }

      /* Map new IO address */

      s->PCI0IO_start = s->regs[GT_PCI0IOLD] << 21;

      s->PCI0IO_length = ((s->regs[GT_PCI0IOHD] + 1) - (s->regs[GT_PCI0IOLD] & 0x7f)) << 21;

      isa_mem_base = s->PCI0IO_start;

      isa_mmio_init(s->PCI0IO_start, s->PCI0IO_length);

    }

}
