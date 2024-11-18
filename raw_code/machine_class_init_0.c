static void machine_class_init(ObjectClass *oc, void *data)

{

    MachineClass *mc = MACHINE_CLASS(oc);

    QEMUMachine *qm = data;

    mc->name = qm->name;

    mc->desc = qm->desc;

    mc->init = qm->init;

    mc->kvm_type = qm->kvm_type;

    mc->block_default_type = qm->block_default_type;

    mc->max_cpus = qm->max_cpus;

    mc->no_sdcard = qm->no_sdcard;

    mc->has_dynamic_sysbus = qm->has_dynamic_sysbus;

    mc->is_default = qm->is_default;

    mc->default_machine_opts = qm->default_machine_opts;

    mc->default_boot_order = qm->default_boot_order;

}
