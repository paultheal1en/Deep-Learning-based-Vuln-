static void ppc_cpu_class_init(ObjectClass *oc, void *data)

{

    PowerPCCPUClass *pcc = POWERPC_CPU_CLASS(oc);

    CPUClass *cc = CPU_CLASS(oc);

    DeviceClass *dc = DEVICE_CLASS(oc);



    pcc->parent_realize = dc->realize;

    dc->realize = ppc_cpu_realizefn;

    dc->unrealize = ppc_cpu_unrealizefn;



    pcc->parent_reset = cc->reset;

    cc->reset = ppc_cpu_reset;



    cc->class_by_name = ppc_cpu_class_by_name;

    cc->do_interrupt = ppc_cpu_do_interrupt;

    cc->dump_state = ppc_cpu_dump_state;

    cc->dump_statistics = ppc_cpu_dump_statistics;

    cc->set_pc = ppc_cpu_set_pc;

    cc->gdb_read_register = ppc_cpu_gdb_read_register;

    cc->gdb_write_register = ppc_cpu_gdb_write_register;

#ifndef CONFIG_USER_ONLY

    cc->get_phys_page_debug = ppc_cpu_get_phys_page_debug;

    cc->vmsd = &vmstate_ppc_cpu;








    cc->gdb_num_core_regs = 71;


    cc->gdb_core_xml_file = "power64-core.xml";

#else

    cc->gdb_core_xml_file = "power-core.xml";


}