static void s390_ccw_realize(S390CCWDevice *cdev, char *sysfsdev, Error **errp)

{

    CcwDevice *ccw_dev = CCW_DEVICE(cdev);

    CCWDeviceClass *ck = CCW_DEVICE_GET_CLASS(ccw_dev);

    DeviceState *parent = DEVICE(ccw_dev);

    BusState *qbus = qdev_get_parent_bus(parent);

    VirtualCssBus *cbus = VIRTUAL_CSS_BUS(qbus);

    SubchDev *sch;

    int ret;

    Error *err = NULL;



    s390_ccw_get_dev_info(cdev, sysfsdev, &err);

    if (err) {

        goto out_err_propagate;

    }



    sch = css_create_sch(ccw_dev->devno, false, cbus->squash_mcss, &err);

    if (!sch) {

        goto out_mdevid_free;

    }

    sch->driver_data = cdev;

    sch->do_subchannel_work = do_subchannel_work_passthrough;



    ccw_dev->sch = sch;

    ret = css_sch_build_schib(sch, &cdev->hostid);

    if (ret) {

        error_setg_errno(&err, -ret, "%s: Failed to build initial schib",

                         __func__);

        goto out_err;

    }



    ck->realize(ccw_dev, &err);

    if (err) {

        goto out_err;

    }



    css_generate_sch_crws(sch->cssid, sch->ssid, sch->schid,

                          parent->hotplugged, 1);

    return;



out_err:

    css_subch_assign(sch->cssid, sch->ssid, sch->schid, sch->devno, NULL);

    ccw_dev->sch = NULL;

    g_free(sch);

out_mdevid_free:

    g_free(cdev->mdevid);

out_err_propagate:

    error_propagate(errp, err);

}
