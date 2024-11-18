static void s390_pcihost_hot_unplug(HotplugHandler *hotplug_dev,

                                    DeviceState *dev, Error **errp)

{

    PCIDevice *pci_dev = NULL;

    PCIBus *bus;

    int32_t devfn;

    S390PCIBusDevice *pbdev = NULL;

    S390pciState *s = s390_get_phb();



    if (object_dynamic_cast(OBJECT(dev), TYPE_PCI_BRIDGE)) {

        error_setg(errp, "PCI bridge hot unplug currently not supported");

        return;

    } else if (object_dynamic_cast(OBJECT(dev), TYPE_PCI_DEVICE)) {

        pci_dev = PCI_DEVICE(dev);



        QTAILQ_FOREACH(pbdev, &s->zpci_devs, link) {

            if (pbdev->pdev == pci_dev) {

                break;

            }

        }

        assert(pbdev != NULL);

    } else if (object_dynamic_cast(OBJECT(dev), TYPE_S390_PCI_DEVICE)) {

        pbdev = S390_PCI_DEVICE(dev);

        pci_dev = pbdev->pdev;

    }



    switch (pbdev->state) {

    case ZPCI_FS_RESERVED:

        goto out;

    case ZPCI_FS_STANDBY:

        break;

    default:

        s390_pci_generate_plug_event(HP_EVENT_DECONFIGURE_REQUEST,

                                     pbdev->fh, pbdev->fid);

        pbdev->release_timer = timer_new_ns(QEMU_CLOCK_VIRTUAL,

                                            s390_pcihost_timer_cb,

                                            pbdev);

        timer_mod(pbdev->release_timer,

                  qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) + HOT_UNPLUG_TIMEOUT);

        return;

    }



    if (pbdev->release_timer && timer_pending(pbdev->release_timer)) {

        timer_del(pbdev->release_timer);

        timer_free(pbdev->release_timer);

        pbdev->release_timer = NULL;

    }



    s390_pci_generate_plug_event(HP_EVENT_STANDBY_TO_RESERVED,

                                 pbdev->fh, pbdev->fid);

    bus = pci_dev->bus;

    devfn = pci_dev->devfn;

    object_unparent(OBJECT(pci_dev));

    s390_pci_msix_free(pbdev);

    s390_pci_iommu_free(s, bus, devfn);

    pbdev->pdev = NULL;

    pbdev->state = ZPCI_FS_RESERVED;

out:

    pbdev->fid = 0;

    QTAILQ_REMOVE(&s->zpci_devs, pbdev, link);

    g_hash_table_remove(s->zpci_table, &pbdev->idx);

    object_unparent(OBJECT(pbdev));

}
