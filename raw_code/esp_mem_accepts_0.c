static bool esp_mem_accepts(void *opaque, target_phys_addr_t addr,

                            unsigned size, bool is_write)

{

    return (size == 1) || (is_write && size == 4);

}
