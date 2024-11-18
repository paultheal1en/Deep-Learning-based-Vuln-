static void mb_cpu_class_init(ObjectClass *oc, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(oc);
    CPUClass *cc = CPU_CLASS(oc);
    MicroBlazeCPUClass *mcc = MICROBLAZE_CPU_CLASS(oc);
    mcc->parent_realize = dc->realize;
    dc->realize = mb_cpu_realizefn;
    mcc->parent_reset = cc->reset;
    cc->reset = mb_cpu_reset;
    cc->has_work = mb_cpu_has_work;
    cc->do_interrupt = mb_cpu_do_interrupt;
    cc->cpu_exec_interrupt = mb_cpu_exec_interrupt;
    cc->dump_state = mb_cpu_dump_state;
    cc->set_pc = mb_cpu_set_pc;
    cc->gdb_read_register = mb_cpu_gdb_read_register;
    cc->gdb_write_register = mb_cpu_gdb_write_register;
#ifdef CONFIG_USER_ONLY
    cc->handle_mmu_fault = mb_cpu_handle_mmu_fault;
#else
    cc->do_unassigned_access = mb_cpu_unassigned_access;
    cc->get_phys_page_debug = mb_cpu_get_phys_page_debug;
#endif
    dc->vmsd = &vmstate_mb_cpu;
    dc->props = mb_properties;
    cc->gdb_num_core_regs = 32 + 5;
    cc->disas_set_info = mb_disas_set_info;
}