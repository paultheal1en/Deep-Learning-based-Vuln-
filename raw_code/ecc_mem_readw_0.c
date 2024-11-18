static uint32_t ecc_mem_readw(void *opaque, target_phys_addr_t addr)

{

    printf("ECC: Unsupported read 0x" TARGET_FMT_plx " 0000\n", addr);

    return 0;

}
