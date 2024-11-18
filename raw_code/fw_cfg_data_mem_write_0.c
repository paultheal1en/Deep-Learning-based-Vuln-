static void fw_cfg_data_mem_write(void *opaque, target_phys_addr_t addr,

                                  uint64_t value, unsigned size)

{

    fw_cfg_write(opaque, (uint8_t)value);

}
