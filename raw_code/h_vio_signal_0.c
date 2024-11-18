static target_ulong h_vio_signal(CPUState *env, sPAPREnvironment *spapr,

                                 target_ulong opcode,

                                 target_ulong *args)

{

    target_ulong reg = args[0];

    target_ulong mode = args[1];

    VIOsPAPRDevice *dev = spapr_vio_find_by_reg(spapr->vio_bus, reg);

    VIOsPAPRDeviceInfo *info;



    if (!dev) {

        return H_PARAMETER;

    }



    info = (VIOsPAPRDeviceInfo *)qdev_get_info(&dev->qdev);



    if (mode & ~info->signal_mask) {

        return H_PARAMETER;

    }



    dev->signal_state = mode;



    return H_SUCCESS;

}
