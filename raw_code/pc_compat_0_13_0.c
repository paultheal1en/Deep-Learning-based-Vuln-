static void pc_compat_0_13(MachineState *machine)

{

    pc_compat_1_2(machine);

    kvmclock_enabled = false;

}
