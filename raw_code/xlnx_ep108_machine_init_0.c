static void xlnx_ep108_machine_init(MachineClass *mc)

{

    mc->desc = "Xilinx ZynqMP EP108 board";

    mc->init = xlnx_ep108_init;

    mc->block_default_type = IF_IDE;

    mc->units_per_default_bus = 1;

    mc->ignore_memory_transaction_failures = true;

}
