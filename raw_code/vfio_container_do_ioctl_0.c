static int vfio_container_do_ioctl(AddressSpace *as, int32_t groupid,

                                   int req, void *param)

{

    VFIOGroup *group;

    VFIOContainer *container;

    int ret = -1;



    group = vfio_get_group(groupid, as);

    if (!group) {

        error_report("vfio: group %d not registered", groupid);

        return ret;

    }



    container = group->container;

    if (group->container) {

        ret = ioctl(container->fd, req, param);

        if (ret < 0) {

            error_report("vfio: failed to ioctl %d to container: ret=%d, %s",

                         _IOC_NR(req) - VFIO_BASE, ret, strerror(errno));

        }

    }



    vfio_put_group(group);



    return ret;

}
