static void vm_change_state_handler(void *opaque, int running,

                                    RunState state)

{

    GICv3ITSState *s = (GICv3ITSState *)opaque;

    Error *err = NULL;

    int ret;



    if (running) {

        return;

    }



    ret = kvm_device_access(s->dev_fd, KVM_DEV_ARM_VGIC_GRP_CTRL,

                            KVM_DEV_ARM_ITS_SAVE_TABLES, NULL, true, &err);

    if (err) {

        error_report_err(err);

    }

    if (ret < 0 && ret != -EFAULT) {

        abort();

    }

}
