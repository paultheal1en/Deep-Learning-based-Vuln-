static void fw_cfg_data_mem_write(void *opaque, hwaddr addr,

                                  uint64_t value, unsigned size)

{

    fw_cfg_write(opaque, (uint8_t)value);

}
