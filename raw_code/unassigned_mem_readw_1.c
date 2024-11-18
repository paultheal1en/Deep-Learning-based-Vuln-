static uint32_t unassigned_mem_readw(void *opaque, target_phys_addr_t addr)

{

#ifdef DEBUG_UNASSIGNED

    printf("Unassigned mem read " TARGET_FMT_plx "\n", addr);

#endif

#if defined(TARGET_ALPHA) || defined(TARGET_SPARC) || defined(TARGET_MICROBLAZE)

    do_unassigned_access(addr, 0, 0, 0, 2);

#endif

    return 0;

}
