static void x86_cpu_unrealizefn(DeviceState *dev, Error **errp)

{

    X86CPU *cpu = X86_CPU(dev);





#ifndef CONFIG_USER_ONLY

    cpu_remove_sync(CPU(dev));

    qemu_unregister_reset(x86_cpu_machine_reset_cb, dev);

#endif



    if (cpu->apic_state) {

        object_unparent(OBJECT(cpu->apic_state));

        cpu->apic_state = NULL;

    }



    xcc->parent_unrealize(dev, &local_err);

    if (local_err != NULL) {

        error_propagate(errp, local_err);

        return;

    }

}