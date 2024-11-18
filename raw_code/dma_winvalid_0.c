dma_winvalid (void *opaque, target_phys_addr_t addr, uint32_t value)

{

        hw_error("Unsupported short waccess. reg=" TARGET_FMT_plx "\n", addr);

}
