static void pc_compat_2_2(MachineState *machine)

{

    pc_compat_2_3(machine);

    rsdp_in_ram = false;

    machine->suppress_vmdesc = true;

}
