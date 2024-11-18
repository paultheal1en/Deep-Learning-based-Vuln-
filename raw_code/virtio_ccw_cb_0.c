static int virtio_ccw_cb(SubchDev *sch, CCW1 ccw)

{

    int ret;

    VirtioRevInfo revinfo;

    uint8_t status;

    VirtioFeatDesc features;

    void *config;

    hwaddr indicators;

    VqConfigBlock vq_config;

    VirtioCcwDevice *dev = sch->driver_data;

    VirtIODevice *vdev = virtio_ccw_get_vdev(sch);

    bool check_len;

    int len;

    hwaddr hw_len;

    VirtioThinintInfo *thinint;



    if (!dev) {

        return -EINVAL;

    }



    trace_virtio_ccw_interpret_ccw(sch->cssid, sch->ssid, sch->schid,

                                   ccw.cmd_code);

    check_len = !((ccw.flags & CCW_FLAG_SLI) && !(ccw.flags & CCW_FLAG_DC));



    if (dev->force_revision_1 && dev->revision < 0 &&

        ccw.cmd_code != CCW_CMD_SET_VIRTIO_REV) {

        /*

         * virtio-1 drivers must start with negotiating to a revision >= 1,

         * so post a command reject for all other commands

         */

        return -ENOSYS;

    }



    /* Look at the command. */

    switch (ccw.cmd_code) {

    case CCW_CMD_SET_VQ:

        ret = virtio_ccw_handle_set_vq(sch, ccw, check_len, dev->revision < 1);

        break;

    case CCW_CMD_VDEV_RESET:

        virtio_ccw_reset_virtio(dev, vdev);

        ret = 0;

        break;

    case CCW_CMD_READ_FEAT:

        if (check_len) {

            if (ccw.count != sizeof(features)) {

                ret = -EINVAL;

                break;

            }

        } else if (ccw.count < sizeof(features)) {

            /* Can't execute command. */

            ret = -EINVAL;

            break;

        }

        if (!ccw.cda) {

            ret = -EFAULT;

        } else {

            VirtioDeviceClass *vdc = VIRTIO_DEVICE_GET_CLASS(vdev);



            features.index = address_space_ldub(&address_space_memory,

                                                ccw.cda

                                                + sizeof(features.features),

                                                MEMTXATTRS_UNSPECIFIED,

                                                NULL);

            if (features.index == 0) {

                if (dev->revision >= 1) {

                    /* Don't offer legacy features for modern devices. */

                    features.features = (uint32_t)

                        (vdev->host_features & ~vdc->legacy_features);

                } else {

                    features.features = (uint32_t)vdev->host_features;

                }

            } else if ((features.index == 1) && (dev->revision >= 1)) {

                /*

                 * Only offer feature bits beyond 31 if the guest has

                 * negotiated at least revision 1.

                 */

                features.features = (uint32_t)(vdev->host_features >> 32);

            } else {

                /* Return zeroes if the guest supports more feature bits. */

                features.features = 0;

            }

            address_space_stl_le(&address_space_memory, ccw.cda,

                                 features.features, MEMTXATTRS_UNSPECIFIED,

                                 NULL);

            sch->curr_status.scsw.count = ccw.count - sizeof(features);

            ret = 0;

        }

        break;

    case CCW_CMD_WRITE_FEAT:

        if (check_len) {

            if (ccw.count != sizeof(features)) {

                ret = -EINVAL;

                break;

            }

        } else if (ccw.count < sizeof(features)) {

            /* Can't execute command. */

            ret = -EINVAL;

            break;

        }

        if (!ccw.cda) {

            ret = -EFAULT;

        } else {

            features.index = address_space_ldub(&address_space_memory,

                                                ccw.cda

                                                + sizeof(features.features),

                                                MEMTXATTRS_UNSPECIFIED,

                                                NULL);

            features.features = address_space_ldl_le(&address_space_memory,

                                                     ccw.cda,

                                                     MEMTXATTRS_UNSPECIFIED,

                                                     NULL);

            if (features.index == 0) {

                virtio_set_features(vdev,

                                    (vdev->guest_features & 0xffffffff00000000ULL) |

                                    features.features);

            } else if ((features.index == 1) && (dev->revision >= 1)) {

                /*

                 * If the guest did not negotiate at least revision 1,

                 * we did not offer it any feature bits beyond 31. Such a

                 * guest passing us any bit here is therefore buggy.

                 */

                virtio_set_features(vdev,

                                    (vdev->guest_features & 0x00000000ffffffffULL) |

                                    ((uint64_t)features.features << 32));

            } else {

                /*

                 * If the guest supports more feature bits, assert that it

                 * passes us zeroes for those we don't support.

                 */

                if (features.features) {

                    fprintf(stderr, "Guest bug: features[%i]=%x (expected 0)\n",

                            features.index, features.features);

                    /* XXX: do a unit check here? */

                }

            }

            sch->curr_status.scsw.count = ccw.count - sizeof(features);

            ret = 0;

        }

        break;

    case CCW_CMD_READ_CONF:

        if (check_len) {

            if (ccw.count > vdev->config_len) {

                ret = -EINVAL;

                break;

            }

        }

        len = MIN(ccw.count, vdev->config_len);

        if (!ccw.cda) {

            ret = -EFAULT;

        } else {

            virtio_bus_get_vdev_config(&dev->bus, vdev->config);

            cpu_physical_memory_write(ccw.cda, vdev->config, len);

            sch->curr_status.scsw.count = ccw.count - len;

            ret = 0;

        }

        break;

    case CCW_CMD_WRITE_CONF:

        if (check_len) {

            if (ccw.count > vdev->config_len) {

                ret = -EINVAL;

                break;

            }

        }

        len = MIN(ccw.count, vdev->config_len);

        hw_len = len;

        if (!ccw.cda) {

            ret = -EFAULT;

        } else {

            config = cpu_physical_memory_map(ccw.cda, &hw_len, 0);

            if (!config) {

                ret = -EFAULT;

            } else {

                len = hw_len;

                memcpy(vdev->config, config, len);

                cpu_physical_memory_unmap(config, hw_len, 0, hw_len);

                virtio_bus_set_vdev_config(&dev->bus, vdev->config);

                sch->curr_status.scsw.count = ccw.count - len;

                ret = 0;

            }

        }

        break;

    case CCW_CMD_READ_STATUS:

        if (check_len) {

            if (ccw.count != sizeof(status)) {

                ret = -EINVAL;

                break;

            }

        } else if (ccw.count < sizeof(status)) {

            /* Can't execute command. */

            ret = -EINVAL;

            break;

        }

        if (!ccw.cda) {

            ret = -EFAULT;

        } else {

            address_space_stb(&address_space_memory, ccw.cda, vdev->status,

                                        MEMTXATTRS_UNSPECIFIED, NULL);

            sch->curr_status.scsw.count = ccw.count - sizeof(vdev->status);;

            ret = 0;

        }

        break;

    case CCW_CMD_WRITE_STATUS:

        if (check_len) {

            if (ccw.count != sizeof(status)) {

                ret = -EINVAL;

                break;

            }

        } else if (ccw.count < sizeof(status)) {

            /* Can't execute command. */

            ret = -EINVAL;

            break;

        }

        if (!ccw.cda) {

            ret = -EFAULT;

        } else {

            status = address_space_ldub(&address_space_memory, ccw.cda,

                                        MEMTXATTRS_UNSPECIFIED, NULL);

            if (!(status & VIRTIO_CONFIG_S_DRIVER_OK)) {

                virtio_ccw_stop_ioeventfd(dev);

            }

            if (virtio_set_status(vdev, status) == 0) {

                if (vdev->status == 0) {

                    virtio_ccw_reset_virtio(dev, vdev);

                }

                if (status & VIRTIO_CONFIG_S_DRIVER_OK) {

                    virtio_ccw_start_ioeventfd(dev);

                }

                sch->curr_status.scsw.count = ccw.count - sizeof(status);

                ret = 0;

            } else {

                /* Trigger a command reject. */

                ret = -ENOSYS;

            }

        }

        break;

    case CCW_CMD_SET_IND:

        if (check_len) {

            if (ccw.count != sizeof(indicators)) {

                ret = -EINVAL;

                break;

            }

        } else if (ccw.count < sizeof(indicators)) {

            /* Can't execute command. */

            ret = -EINVAL;

            break;

        }

        if (sch->thinint_active) {

            /* Trigger a command reject. */

            ret = -ENOSYS;

            break;

        }

        if (virtio_get_num_queues(vdev) > NR_CLASSIC_INDICATOR_BITS) {

            /* More queues than indicator bits --> trigger a reject */

            ret = -ENOSYS;

            break;

        }

        if (!ccw.cda) {

            ret = -EFAULT;

        } else {

            indicators = address_space_ldq_be(&address_space_memory, ccw.cda,

                                              MEMTXATTRS_UNSPECIFIED, NULL);

            dev->indicators = get_indicator(indicators, sizeof(uint64_t));

            sch->curr_status.scsw.count = ccw.count - sizeof(indicators);

            ret = 0;

        }

        break;

    case CCW_CMD_SET_CONF_IND:

        if (check_len) {

            if (ccw.count != sizeof(indicators)) {

                ret = -EINVAL;

                break;

            }

        } else if (ccw.count < sizeof(indicators)) {

            /* Can't execute command. */

            ret = -EINVAL;

            break;

        }

        if (!ccw.cda) {

            ret = -EFAULT;

        } else {

            indicators = address_space_ldq_be(&address_space_memory, ccw.cda,

                                              MEMTXATTRS_UNSPECIFIED, NULL);

            dev->indicators2 = get_indicator(indicators, sizeof(uint64_t));

            sch->curr_status.scsw.count = ccw.count - sizeof(indicators);

            ret = 0;

        }

        break;

    case CCW_CMD_READ_VQ_CONF:

        if (check_len) {

            if (ccw.count != sizeof(vq_config)) {

                ret = -EINVAL;

                break;

            }

        } else if (ccw.count < sizeof(vq_config)) {

            /* Can't execute command. */

            ret = -EINVAL;

            break;

        }

        if (!ccw.cda) {

            ret = -EFAULT;

        } else {

            vq_config.index = address_space_lduw_be(&address_space_memory,

                                                    ccw.cda,

                                                    MEMTXATTRS_UNSPECIFIED,

                                                    NULL);

            if (vq_config.index >= VIRTIO_QUEUE_MAX) {

                ret = -EINVAL;

                break;

            }

            vq_config.num_max = virtio_queue_get_num(vdev,

                                                     vq_config.index);

            address_space_stw_be(&address_space_memory,

                                 ccw.cda + sizeof(vq_config.index),

                                 vq_config.num_max,

                                 MEMTXATTRS_UNSPECIFIED,

                                 NULL);

            sch->curr_status.scsw.count = ccw.count - sizeof(vq_config);

            ret = 0;

        }

        break;

    case CCW_CMD_SET_IND_ADAPTER:

        if (check_len) {

            if (ccw.count != sizeof(*thinint)) {

                ret = -EINVAL;

                break;

            }

        } else if (ccw.count < sizeof(*thinint)) {

            /* Can't execute command. */

            ret = -EINVAL;

            break;

        }

        len = sizeof(*thinint);

        hw_len = len;

        if (!ccw.cda) {

            ret = -EFAULT;

        } else if (dev->indicators && !sch->thinint_active) {

            /* Trigger a command reject. */

            ret = -ENOSYS;

        } else {

            thinint = cpu_physical_memory_map(ccw.cda, &hw_len, 0);

            if (!thinint) {

                ret = -EFAULT;

            } else {

                uint64_t ind_bit = ldq_be_p(&thinint->ind_bit);



                len = hw_len;

                dev->summary_indicator =

                    get_indicator(ldq_be_p(&thinint->summary_indicator),

                                  sizeof(uint8_t));

                dev->indicators =

                    get_indicator(ldq_be_p(&thinint->device_indicator),

                                  ind_bit / 8 + 1);

                dev->thinint_isc = thinint->isc;

                dev->routes.adapter.ind_offset = ind_bit;

                dev->routes.adapter.summary_offset = 7;

                cpu_physical_memory_unmap(thinint, hw_len, 0, hw_len);

                dev->routes.adapter.adapter_id = css_get_adapter_id(

                                                 CSS_IO_ADAPTER_VIRTIO,

                                                 dev->thinint_isc);

                sch->thinint_active = ((dev->indicators != NULL) &&

                                       (dev->summary_indicator != NULL));

                sch->curr_status.scsw.count = ccw.count - len;

                ret = 0;

            }

        }

        break;

    case CCW_CMD_SET_VIRTIO_REV:

        len = sizeof(revinfo);

        if (ccw.count < len) {

            ret = -EINVAL;

            break;

        }

        if (!ccw.cda) {

            ret = -EFAULT;

            break;

        }

        revinfo.revision =

            address_space_lduw_be(&address_space_memory, ccw.cda,

                                  MEMTXATTRS_UNSPECIFIED, NULL);

        revinfo.length =

            address_space_lduw_be(&address_space_memory,

                                  ccw.cda + sizeof(revinfo.revision),

                                  MEMTXATTRS_UNSPECIFIED, NULL);

        if (ccw.count < len + revinfo.length ||

            (check_len && ccw.count > len + revinfo.length)) {

            ret = -EINVAL;

            break;

        }

        /*

         * Once we start to support revisions with additional data, we'll

         * need to fetch it here. Nothing to do for now, though.

         */

        if (dev->revision >= 0 ||

            revinfo.revision > virtio_ccw_rev_max(dev) ||

            (dev->force_revision_1 && !revinfo.revision)) {

            ret = -ENOSYS;

            break;

        }

        ret = 0;

        dev->revision = revinfo.revision;

        break;

    default:

        ret = -ENOSYS;

        break;

    }

    return ret;

}
