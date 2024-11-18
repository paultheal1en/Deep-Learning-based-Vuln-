static void vga_mm_init(VGAState *s, target_phys_addr_t vram_base,

                        target_phys_addr_t ctrl_base, int it_shift)

{

    int s_ioport_ctrl, vga_io_memory;



    s->it_shift = it_shift;

    s_ioport_ctrl = cpu_register_io_memory(0, vga_mm_read_ctrl, vga_mm_write_ctrl, s);

    vga_io_memory = cpu_register_io_memory(0, vga_mem_read, vga_mem_write, s);



    register_savevm("vga", 0, 2, vga_save, vga_load, s);



    cpu_register_physical_memory(ctrl_base, 0x100000, s_ioport_ctrl);

    s->bank_offset = 0;

    cpu_register_physical_memory(vram_base + 0x000a0000, 0x20000, vga_io_memory);


}