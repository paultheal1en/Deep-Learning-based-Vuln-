uint64_t ldq_phys(target_phys_addr_t addr)

{

    return ldq_phys_internal(addr, DEVICE_NATIVE_ENDIAN);

}
