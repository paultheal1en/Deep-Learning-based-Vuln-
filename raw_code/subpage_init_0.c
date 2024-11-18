static subpage_t *subpage_init(target_phys_addr_t base)

{

    subpage_t *mmio;



    mmio = g_malloc0(sizeof(subpage_t));



    mmio->base = base;

    memory_region_init_io(&mmio->iomem, &subpage_ops, mmio,

                          "subpage", TARGET_PAGE_SIZE);

    mmio->iomem.subpage = true;

#if defined(DEBUG_SUBPAGE)

    printf("%s: %p base " TARGET_FMT_plx " len %08x %d\n", __func__,

           mmio, base, TARGET_PAGE_SIZE, subpage_memory);

#endif

    subpage_register(mmio, 0, TARGET_PAGE_SIZE-1, phys_section_unassigned);



    return mmio;

}
