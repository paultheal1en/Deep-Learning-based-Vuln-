static int spapr_vio_busdev_init(DeviceState *qdev, DeviceInfo *qinfo)

{

    VIOsPAPRDeviceInfo *info = (VIOsPAPRDeviceInfo *)qinfo;

    VIOsPAPRDevice *dev = (VIOsPAPRDevice *)qdev;

    char *id;

    int ret;



    ret = spapr_vio_check_reg(dev, info);

    if (ret) {

        return ret;

    }



    /* Don't overwrite ids assigned on the command line */

    if (!dev->qdev.id) {

        id = vio_format_dev_name(dev);

        if (!id) {

            return -1;

        }

        dev->qdev.id = id;

    }



    dev->qirq = spapr_allocate_irq(dev->vio_irq_num, &dev->vio_irq_num);

    if (!dev->qirq) {

        return -1;

    }



    rtce_init(dev);



    return info->init(dev);

}
