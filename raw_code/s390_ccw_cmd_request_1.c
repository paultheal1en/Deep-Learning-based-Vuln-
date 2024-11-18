int s390_ccw_cmd_request(ORB *orb, SCSW *scsw, void *data)

{

    S390CCWDeviceClass *cdc = S390_CCW_DEVICE_GET_CLASS(data);



    if (cdc->handle_request) {

        return cdc->handle_request(orb, scsw, data);

    } else {

        return -ENOSYS;

    }

}
