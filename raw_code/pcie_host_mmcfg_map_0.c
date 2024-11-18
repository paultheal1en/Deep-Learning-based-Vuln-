void pcie_host_mmcfg_map(PCIExpressHost *e, hwaddr addr,

                         uint32_t size)

{

    assert(!(size & (size - 1)));       /* power of 2 */

    assert(size >= PCIE_MMCFG_SIZE_MIN);

    assert(size <= PCIE_MMCFG_SIZE_MAX);

    e->size = size;

    memory_region_init_io(&e->mmio, OBJECT(e), &pcie_mmcfg_ops, e,

                          "pcie-mmcfg", e->size);

    e->base_addr = addr;

    memory_region_add_subregion(get_system_memory(), e->base_addr, &e->mmio);

}
