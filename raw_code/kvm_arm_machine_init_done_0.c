static void kvm_arm_machine_init_done(Notifier *notifier, void *data)

{

    KVMDevice *kd, *tkd;



    memory_listener_unregister(&devlistener);

    QSLIST_FOREACH_SAFE(kd, &kvm_devices_head, entries, tkd) {

        if (kd->kda.addr != -1) {

            if (kvm_vm_ioctl(kvm_state, KVM_ARM_SET_DEVICE_ADDR,

                             &kd->kda) < 0) {

                fprintf(stderr, "KVM_ARM_SET_DEVICE_ADDRESS failed: %s\n",

                        strerror(errno));

                abort();

            }

        }

        memory_region_unref(kd->mr);

        g_free(kd);

    }

}
