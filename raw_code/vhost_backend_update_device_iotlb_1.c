int vhost_backend_update_device_iotlb(struct vhost_dev *dev,

                                             uint64_t iova, uint64_t uaddr,

                                             uint64_t len,

                                             IOMMUAccessFlags perm)

{

    struct vhost_iotlb_msg imsg;



    imsg.iova =  iova;

    imsg.uaddr = uaddr;

    imsg.size = len;

    imsg.type = VHOST_IOTLB_UPDATE;



    switch (perm) {

    case IOMMU_RO:

        imsg.perm = VHOST_ACCESS_RO;

        break;

    case IOMMU_WO:

        imsg.perm = VHOST_ACCESS_WO;

        break;

    case IOMMU_RW:

        imsg.perm = VHOST_ACCESS_RW;

        break;

    default:

        return -EINVAL;

    }



    return dev->vhost_ops->vhost_send_device_iotlb_msg(dev, &imsg);

}
