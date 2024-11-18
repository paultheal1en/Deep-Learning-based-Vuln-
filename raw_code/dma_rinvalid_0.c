static uint32_t dma_rinvalid (void *opaque, target_phys_addr_t addr)

{

        hw_error("Unsupported short raccess. reg=" TARGET_FMT_plx "\n", addr);

        return 0;

}
