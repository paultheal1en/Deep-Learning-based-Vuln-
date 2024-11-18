void cuda_init (int *cuda_mem_index, qemu_irq irq)

{

    struct tm tm;

    CUDAState *s = &cuda_state;



    s->irq = irq;



    s->timers[0].index = 0;

    s->timers[0].timer = qemu_new_timer(vm_clock, cuda_timer1, s);



    s->timers[1].index = 1;



    qemu_get_timedate(&tm, RTC_OFFSET);

    s->tick_offset = mktimegm(&tm);



    s->adb_poll_timer = qemu_new_timer(vm_clock, cuda_adb_poll, s);

    *cuda_mem_index = cpu_register_io_memory(0, cuda_read, cuda_write, s);

    register_savevm("cuda", -1, 1, cuda_save, cuda_load, s);

    qemu_register_reset(cuda_reset, s);

    cuda_reset(s);

}
