static uint32_t hpet_ram_readw(void *opaque, target_phys_addr_t addr)

{

    printf("qemu: hpet_read w at %" PRIx64 "\n", addr);

    return 0;

}
