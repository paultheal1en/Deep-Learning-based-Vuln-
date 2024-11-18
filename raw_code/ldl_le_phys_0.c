uint32_t ldl_le_phys(target_phys_addr_t addr)

{

    return ldl_phys_internal(addr, DEVICE_LITTLE_ENDIAN);

}
