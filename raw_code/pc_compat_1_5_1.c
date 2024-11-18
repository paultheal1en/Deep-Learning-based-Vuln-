static void pc_compat_1_5(QEMUMachineInitArgs *args)

{

    pc_compat_1_6(args);

    has_pvpanic = true;

}
