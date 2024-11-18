static void uc32_cpu_class_init(ObjectClass *oc, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(oc);
    CPUClass *cc = CPU_CLASS(oc);
    UniCore32CPUClass *ucc = UNICORE32_CPU_CLASS(oc);
    ucc->parent_realize = dc->realize;
    dc->realize = uc32_cpu_realizefn;
    cc->class_by_name = uc32_cpu_class_by_name;
    cc->has_work = uc32_cpu_has_work;
    cc->do_interrupt = uc32_cpu_do_interrupt;
    cc->cpu_exec_interrupt = uc32_cpu_exec_interrupt;
    cc->dump_state = uc32_cpu_dump_state;
    cc->set_pc = uc32_cpu_set_pc;
#ifdef CONFIG_USER_ONLY
    cc->handle_mmu_fault = uc32_cpu_handle_mmu_fault;
#else
    cc->get_phys_page_debug = uc32_cpu_get_phys_page_debug;
#endif
    dc->vmsd = &vmstate_uc32_cpu;
}