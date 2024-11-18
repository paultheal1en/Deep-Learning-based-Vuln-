static void omap_mpui_init(MemoryRegion *memory, target_phys_addr_t base,

                struct omap_mpu_state_s *mpu)

{

    memory_region_init_io(&mpu->mpui_iomem, &omap_mpui_ops, mpu,

                          "omap-mpui", 0x100);

    memory_region_add_subregion(memory, base, &mpu->mpui_iomem);



    omap_mpui_reset(mpu);

}
