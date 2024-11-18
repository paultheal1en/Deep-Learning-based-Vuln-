static uint32_t ecc_mem_readb(void *opaque, target_phys_addr_t addr)

{

    printf("ECC: Unsupported read 0x" TARGET_FMT_plx " 00\n", addr);

    return 0;

}
