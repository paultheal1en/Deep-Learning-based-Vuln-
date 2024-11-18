static int spapr_vio_check_reg(VIOsPAPRDevice *sdev, VIOsPAPRDeviceInfo *info)

{

    VIOsPAPRDevice *other_sdev;

    DeviceState *qdev;

    VIOsPAPRBus *sbus;



    sbus = DO_UPCAST(VIOsPAPRBus, bus, sdev->qdev.parent_bus);



    /*

     * Check two device aren't given clashing addresses by the user (or some

     * other mechanism). We have to open code this because we have to check

     * for matches with devices other than us.

     */

    QTAILQ_FOREACH(qdev, &sbus->bus.children, sibling) {

        other_sdev = DO_UPCAST(VIOsPAPRDevice, qdev, qdev);



        if (other_sdev != sdev && other_sdev->reg == sdev->reg) {

            fprintf(stderr, "vio: %s and %s devices conflict at address %#x\n",

                    info->qdev.name, other_sdev->qdev.info->name, sdev->reg);

            return -EEXIST;

        }

    }



    return 0;

}
