static void tricore_cpu_class_init(ObjectClass *c, void *data)

{

    TriCoreCPUClass *mcc = TRICORE_CPU_CLASS(c);

    CPUClass *cc = CPU_CLASS(c);

    DeviceClass *dc = DEVICE_CLASS(c);



    mcc->parent_realize = dc->realize;

    dc->realize = tricore_cpu_realizefn;



    mcc->parent_reset = cc->reset;

    cc->reset = tricore_cpu_reset;

    cc->class_by_name = tricore_cpu_class_by_name;

    cc->has_work = tricore_cpu_has_work;



    cc->dump_state = tricore_cpu_dump_state;

    cc->set_pc = tricore_cpu_set_pc;

    cc->synchronize_from_tb = tricore_cpu_synchronize_from_tb;


}