static void pci_bridge_cleanup_alias(MemoryRegion *alias,

                                     MemoryRegion *parent_space)

{

    memory_region_del_subregion(parent_space, alias);

    memory_region_destroy(alias);

}
