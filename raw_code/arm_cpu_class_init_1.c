static void arm_cpu_class_init(ObjectClass *oc, void *data)
{
    ARMCPUClass *acc = ARM_CPU_CLASS(oc);
    CPUClass *cc = CPU_CLASS(acc);
    DeviceClass *dc = DEVICE_CLASS(oc);
    acc->parent_realize = dc->realize;
    dc->realize = arm_cpu_realizefn;
    dc->props = arm_cpu_properties;
    acc->parent_reset = cc->reset;
    cc->reset = arm_cpu_reset;
    cc->class_by_name = arm_cpu_class_by_name;
    cc->has_work = arm_cpu_has_work;
    cc->cpu_exec_interrupt = arm_cpu_exec_interrupt;
    cc->dump_state = arm_cpu_dump_state;
    cc->set_pc = arm_cpu_set_pc;
    cc->gdb_read_register = arm_cpu_gdb_read_register;
    cc->gdb_write_register = arm_cpu_gdb_write_register;
#ifdef CONFIG_USER_ONLY
    cc->handle_mmu_fault = arm_cpu_handle_mmu_fault;
#else
    cc->do_interrupt = arm_cpu_do_interrupt;
    cc->get_phys_page_debug = arm_cpu_get_phys_page_debug;
    cc->vmsd = &vmstate_arm_cpu;
    cc->virtio_is_big_endian = arm_cpu_is_big_endian;
#endif
    cc->gdb_num_core_regs = 26;
    cc->gdb_core_xml_file = "arm-core.xml";
    cc->gdb_stop_before_watchpoint = true;
    cc->debug_excp_handler = arm_debug_excp_handler;
    cc->disas_set_info = arm_disas_set_info;
}