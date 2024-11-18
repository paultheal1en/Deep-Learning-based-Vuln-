static void ccw_machine_2_9_class_options(MachineClass *mc)

{

    S390CcwMachineClass *s390mc = S390_MACHINE_CLASS(mc);



    s390mc->gs_allowed = false;

    ccw_machine_2_10_class_options(mc);

    SET_MACHINE_COMPAT(mc, CCW_COMPAT_2_9);

    s390mc->css_migration_enabled = false;

}
