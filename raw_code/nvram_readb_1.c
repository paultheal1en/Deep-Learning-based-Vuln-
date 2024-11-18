static uint32_t nvram_readb (void *opaque, target_phys_addr_t addr)

{

    ds1225y_t *NVRAM = opaque;

    int64_t pos;



    pos = addr - NVRAM->mem_base;

    if (addr >= NVRAM->capacity)

        addr -= NVRAM->capacity;



    if (!ds1225y_set_to_mode(NVRAM, readmode, "rb"))

        return 0;

    qemu_fseek(NVRAM->file, pos, SEEK_SET);

    return (uint32_t)qemu_get_byte(NVRAM->file);

}
