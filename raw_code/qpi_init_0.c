static void qpi_init(void)

{

    kqemu_comm_base = 0xff000000 | 1;

    qpi_io_memory = cpu_register_io_memory(

                                           qpi_mem_read, 

                                           qpi_mem_write, NULL);

    cpu_register_physical_memory(kqemu_comm_base & ~0xfff, 

                                 0x1000, qpi_io_memory);

}
