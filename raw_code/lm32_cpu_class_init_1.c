static void lm32_cpu_class_init(ObjectClass *oc, void *data)
{
    LM32CPUClass *lcc = LM32_CPU_CLASS(oc);
    CPUClass *cc = CPU_CLASS(oc);
    DeviceClass *dc = DEVICE_CLASS(oc);
    lcc->parent_realize = dc->realize;
    dc->realize = lm32_cpu_realizefn;
    lcc->parent_reset = cc->reset;
    cc->reset = lm32_cpu_reset;
    cc->class_by_name = lm32_cpu_class_by_name;
    cc->has_work = lm32_cpu_has_work;
    cc->do_interrupt = lm32_cpu_do_interrupt;
    cc->cpu_exec_interrupt = lm32_cpu_exec_interrupt;
    cc->dump_state = lm32_cpu_dump_state;
    cc->set_pc = lm32_cpu_set_pc;
    cc->gdb_read_register = lm32_cpu_gdb_read_register;
    cc->gdb_write_register = lm32_cpu_gdb_write_register;
#ifdef CONFIG_USER_ONLY
    cc->handle_mmu_fault = lm32_cpu_handle_mmu_fault;
#else
    cc->get_phys_page_debug = lm32_cpu_get_phys_page_debug;
    cc->vmsd = &vmstate_lm32_cpu;
#endif
    cc->gdb_num_core_regs = 32 + 7;
    cc->gdb_stop_before_watchpoint = true;
    cc->debug_excp_handler = lm32_debug_excp_handler;
}