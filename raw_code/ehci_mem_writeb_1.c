static void ehci_mem_writeb(void *ptr, target_phys_addr_t addr, uint32_t val)

{

    fprintf(stderr, "EHCI doesn't handle byte writes to MMIO\n");

    exit(1);

}
