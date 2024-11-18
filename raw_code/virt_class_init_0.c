static void virt_class_init(ObjectClass *oc, void *data)

{

    MachineClass *mc = MACHINE_CLASS(oc);



    mc->desc = "ARM Virtual Machine",

    mc->init = machvirt_init;

    mc->max_cpus = 8;

    mc->has_dynamic_sysbus = true;

    mc->block_default_type = IF_VIRTIO;

    mc->no_cdrom = 1;

}
