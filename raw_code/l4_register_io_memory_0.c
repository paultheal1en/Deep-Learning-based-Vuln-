int l4_register_io_memory(CPUReadMemoryFunc * const *mem_read,

                CPUWriteMemoryFunc * const *mem_write, void *opaque)

{

    omap_l4_io_entry[omap_l4_io_entries].mem_read = mem_read;

    omap_l4_io_entry[omap_l4_io_entries].mem_write = mem_write;

    omap_l4_io_entry[omap_l4_io_entries].opaque = opaque;



    return omap_l4_io_entries ++;

}
