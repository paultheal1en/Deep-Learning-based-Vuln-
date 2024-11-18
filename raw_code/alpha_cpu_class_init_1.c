static void alpha_cpu_class_init(ObjectClass *oc, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(oc);

    CPUClass *cc = CPU_CLASS(oc);

    AlphaCPUClass *acc = ALPHA_CPU_CLASS(oc);



    acc->parent_realize = dc->realize;

    dc->realize = alpha_cpu_realizefn;



    cc->class_by_name = alpha_cpu_class_by_name;

    cc->has_work = alpha_cpu_has_work;

    cc->do_interrupt = alpha_cpu_do_interrupt;

    cc->cpu_exec_interrupt = alpha_cpu_exec_interrupt;

    cc->dump_state = alpha_cpu_dump_state;

    cc->set_pc = alpha_cpu_set_pc;

    cc->gdb_read_register = alpha_cpu_gdb_read_register;

    cc->gdb_write_register = alpha_cpu_gdb_write_register;

#ifdef CONFIG_USER_ONLY

    cc->handle_mmu_fault = alpha_cpu_handle_mmu_fault;

#else

    cc->do_unassigned_access = alpha_cpu_unassigned_access;

    cc->do_unaligned_access = alpha_cpu_do_unaligned_access;

    cc->get_phys_page_debug = alpha_cpu_get_phys_page_debug;

    dc->vmsd = &vmstate_alpha_cpu;

#endif

    cc->disas_set_info = alpha_cpu_disas_set_info;



    cc->gdb_num_core_regs = 67;



    /*

     * Reason: alpha_cpu_initfn() calls cpu_exec_init(), which saves

     * the object in cpus -> dangling pointer after final

     * object_unref().

     */

    dc->cannot_destroy_with_object_finalize_yet = true;

}
