static void qemu_rdma_init_one_block(void *host_addr,

    ram_addr_t block_offset, ram_addr_t length, void *opaque)

{

    __qemu_rdma_add_block(opaque, host_addr, block_offset, length);

}
