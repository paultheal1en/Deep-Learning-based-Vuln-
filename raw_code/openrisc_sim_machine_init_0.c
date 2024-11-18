static void openrisc_sim_machine_init(MachineClass *mc)

{

    mc->desc = "or1k simulation";

    mc->init = openrisc_sim_init;

    mc->max_cpus = 1;

    mc->is_default = 1;

}
