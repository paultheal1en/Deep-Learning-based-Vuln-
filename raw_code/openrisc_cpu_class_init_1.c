static void openrisc_cpu_class_init(ObjectClass *oc, void *data)

{

    OpenRISCCPUClass *occ = OPENRISC_CPU_CLASS(oc);

    CPUClass *cc = CPU_CLASS(occ);

    DeviceClass *dc = DEVICE_CLASS(oc);



    occ->parent_realize = dc->realize;

    dc->realize = openrisc_cpu_realizefn;



    occ->parent_reset = cc->reset;

    cc->reset = openrisc_cpu_reset;



    cc->class_by_name = openrisc_cpu_class_by_name;

    cc->has_work = openrisc_cpu_has_work;

    cc->do_interrupt = openrisc_cpu_do_interrupt;

    cc->cpu_exec_interrupt = openrisc_cpu_exec_interrupt;

    cc->dump_state = openrisc_cpu_dump_state;

    cc->set_pc = openrisc_cpu_set_pc;

    cc->gdb_read_register = openrisc_cpu_gdb_read_register;

    cc->gdb_write_register = openrisc_cpu_gdb_write_register;

#ifdef CONFIG_USER_ONLY

    cc->handle_mmu_fault = openrisc_cpu_handle_mmu_fault;

#else

    cc->get_phys_page_debug = openrisc_cpu_get_phys_page_debug;

    dc->vmsd = &vmstate_openrisc_cpu;

#endif

    cc->gdb_num_core_regs = 32 + 3;



    /*

     * Reason: openrisc_cpu_initfn() calls cpu_exec_init(), which saves

     * the object in cpus -> dangling pointer after final

     * object_unref().

     */

    dc->cannot_destroy_with_object_finalize_yet = true;

}
