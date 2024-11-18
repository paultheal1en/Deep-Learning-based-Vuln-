static void isapc_machine_options(MachineClass *m)

{

    pc_common_machine_options(m);

    m->desc = "ISA-only PC";

    m->max_cpus = 1;

}
