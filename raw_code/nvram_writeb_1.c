static void nvram_writeb (void *opaque, target_phys_addr_t addr, uint32_t value)

{

    ds1225y_t *NVRAM = opaque;

    int64_t pos;



    pos = addr - NVRAM->mem_base;

    if (ds1225y_set_to_mode(NVRAM, writemode, "wb"))

    {

        qemu_fseek(NVRAM->file, pos, SEEK_SET);

        qemu_put_byte(NVRAM->file, (int)value);

    }

}
