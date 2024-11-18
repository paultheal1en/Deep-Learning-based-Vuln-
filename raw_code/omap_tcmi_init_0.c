static void omap_tcmi_init(MemoryRegion *memory, target_phys_addr_t base,

                struct omap_mpu_state_s *mpu)

{

    memory_region_init_io(&mpu->tcmi_iomem, &omap_tcmi_ops, mpu,

                          "omap-tcmi", 0x100);

    memory_region_add_subregion(memory, base, &mpu->tcmi_iomem);

    omap_tcmi_reset(mpu);

}
