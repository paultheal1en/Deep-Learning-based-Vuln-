static void vtd_interrupt_remap_table_setup(IntelIOMMUState *s)

{

    uint64_t value = 0;

    value = vtd_get_quad_raw(s, DMAR_IRTA_REG);

    s->intr_size = 1UL << ((value & VTD_IRTA_SIZE_MASK) + 1);

    s->intr_root = value & VTD_IRTA_ADDR_MASK;



    /* TODO: invalidate interrupt entry cache */



    VTD_DPRINTF(CSR, "int remap table addr 0x%"PRIx64 " size %"PRIu32,

                s->intr_root, s->intr_size);

}
