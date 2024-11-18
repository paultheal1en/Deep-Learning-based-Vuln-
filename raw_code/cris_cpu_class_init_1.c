static void cris_cpu_class_init(ObjectClass *oc, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(oc);

    CPUClass *cc = CPU_CLASS(oc);

    CRISCPUClass *ccc = CRIS_CPU_CLASS(oc);



    ccc->parent_realize = dc->realize;

    dc->realize = cris_cpu_realizefn;



    ccc->parent_reset = cc->reset;

    cc->reset = cris_cpu_reset;



    cc->class_by_name = cris_cpu_class_by_name;

    cc->has_work = cris_cpu_has_work;

    cc->do_interrupt = cris_cpu_do_interrupt;

    cc->cpu_exec_interrupt = cris_cpu_exec_interrupt;

    cc->dump_state = cris_cpu_dump_state;

    cc->set_pc = cris_cpu_set_pc;

    cc->gdb_read_register = cris_cpu_gdb_read_register;

    cc->gdb_write_register = cris_cpu_gdb_write_register;

#ifdef CONFIG_USER_ONLY

    cc->handle_mmu_fault = cris_cpu_handle_mmu_fault;

#else

    cc->get_phys_page_debug = cris_cpu_get_phys_page_debug;

    dc->vmsd = &vmstate_cris_cpu;

#endif



    cc->gdb_num_core_regs = 49;

    cc->gdb_stop_before_watchpoint = true;



    cc->disas_set_info = cris_disas_set_info;



    /*

     * Reason: cris_cpu_initfn() calls cpu_exec_init(), which saves

     * the object in cpus -> dangling pointer after final

     * object_unref().

     */

    dc->cannot_destroy_with_object_finalize_yet = true;

}
