ParallelState *parallel_mm_init(target_phys_addr_t base, int it_shift, qemu_irq irq, CharDriverState *chr)

{

    ParallelState *s;

    int io_sw;



    s = qemu_mallocz(sizeof(ParallelState));

    s->irq = irq;

    s->chr = chr;

    s->it_shift = it_shift;

    qemu_register_reset(parallel_reset, s);



    io_sw = cpu_register_io_memory(parallel_mm_read_sw, parallel_mm_write_sw,

                                   s, DEVICE_NATIVE_ENDIAN);

    cpu_register_physical_memory(base, 8 << it_shift, io_sw);

    return s;

}
