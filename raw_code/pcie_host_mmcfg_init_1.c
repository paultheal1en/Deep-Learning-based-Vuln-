void pcie_host_mmcfg_init(PCIExpressHost *e, uint32_t size)

{

    assert(!(size & (size - 1)));       /* power of 2 */

    assert(size >= PCIE_MMCFG_SIZE_MIN);

    assert(size <= PCIE_MMCFG_SIZE_MAX);

    e->size = size;

    memory_region_init_io(&e->mmio, OBJECT(e), &pcie_mmcfg_ops, e,

                          "pcie-mmcfg", e->size);

}
