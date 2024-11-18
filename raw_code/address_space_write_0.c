void address_space_write(AddressSpace *as, target_phys_addr_t addr,

                         const uint8_t *buf, int len)

{

    address_space_rw(as, addr, (uint8_t *)buf, len, true);

}
