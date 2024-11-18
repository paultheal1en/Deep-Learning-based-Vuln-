static void armv7m_nvic_init(SysBusDevice *dev)

{

    nvic_state *s= FROM_SYSBUSGIC(nvic_state, dev);

    CPUState *env;



    env = qdev_get_prop_ptr(&dev->qdev, "cpu");

    gic_init(&s->gic);

    cpu_register_physical_memory(0xe000e000, 0x1000, s->gic.iomemtype);

    s->systick.timer = qemu_new_timer(vm_clock, systick_timer_tick, s);

    if (env->v7m.nvic)

        hw_error("CPU can only have one NVIC\n");

    env->v7m.nvic = s;

    register_savevm("armv7m_nvic", -1, 1, nvic_save, nvic_load, s);

}
