static void pci_msix(void)

{

    QVirtioPCIDevice *dev;

    QPCIBus *bus;

    QVirtQueuePCI *vqpci;

    QGuestAllocator *alloc;

    QVirtioBlkReq req;

    int n_size = TEST_IMAGE_SIZE / 2;

    void *addr;

    uint64_t req_addr;

    uint64_t capacity;

    uint32_t features;

    uint32_t free_head;

    uint8_t status;

    char *data;



    bus = pci_test_start();

    alloc = pc_alloc_init();



    dev = virtio_blk_pci_init(bus, PCI_SLOT);

    qpci_msix_enable(dev->pdev);



    qvirtio_pci_set_msix_configuration_vector(dev, alloc, 0);



    /* MSI-X is enabled */

    addr = dev->addr + VIRTIO_PCI_CONFIG_OFF(true);



    capacity = qvirtio_config_readq(&qvirtio_pci, &dev->vdev,

                                                    (uint64_t)(uintptr_t)addr);

    g_assert_cmpint(capacity, ==, TEST_IMAGE_SIZE / 512);



    features = qvirtio_get_features(&qvirtio_pci, &dev->vdev);

    features = features & ~(QVIRTIO_F_BAD_FEATURE |

                            (1u << VIRTIO_RING_F_INDIRECT_DESC) |

                            (1u << VIRTIO_RING_F_EVENT_IDX) |

                            (1u << VIRTIO_BLK_F_SCSI));

    qvirtio_set_features(&qvirtio_pci, &dev->vdev, features);



    vqpci = (QVirtQueuePCI *)qvirtqueue_setup(&qvirtio_pci, &dev->vdev,

                                                                    alloc, 0);

    qvirtqueue_pci_msix_setup(dev, vqpci, alloc, 1);



    qvirtio_set_driver_ok(&qvirtio_pci, &dev->vdev);



    qmp("{ 'execute': 'block_resize', 'arguments': { 'device': 'drive0', "

                                                    " 'size': %d } }", n_size);



    qvirtio_wait_config_isr(&qvirtio_pci, &dev->vdev, QVIRTIO_BLK_TIMEOUT_US);



    capacity = qvirtio_config_readq(&qvirtio_pci, &dev->vdev,

                                                    (uint64_t)(uintptr_t)addr);

    g_assert_cmpint(capacity, ==, n_size / 512);



    /* Write request */

    req.type = VIRTIO_BLK_T_OUT;

    req.ioprio = 1;

    req.sector = 0;

    req.data = g_malloc0(512);

    strcpy(req.data, "TEST");



    req_addr = virtio_blk_request(alloc, &req, 512);



    g_free(req.data);



    free_head = qvirtqueue_add(&vqpci->vq, req_addr, 16, false, true);

    qvirtqueue_add(&vqpci->vq, req_addr + 16, 512, false, true);

    qvirtqueue_add(&vqpci->vq, req_addr + 528, 1, true, false);

    qvirtqueue_kick(&qvirtio_pci, &dev->vdev, &vqpci->vq, free_head);



    qvirtio_wait_queue_isr(&qvirtio_pci, &dev->vdev, &vqpci->vq,

                           QVIRTIO_BLK_TIMEOUT_US);



    status = readb(req_addr + 528);

    g_assert_cmpint(status, ==, 0);



    guest_free(alloc, req_addr);



    /* Read request */

    req.type = VIRTIO_BLK_T_IN;

    req.ioprio = 1;

    req.sector = 0;

    req.data = g_malloc0(512);



    req_addr = virtio_blk_request(alloc, &req, 512);



    g_free(req.data);



    free_head = qvirtqueue_add(&vqpci->vq, req_addr, 16, false, true);

    qvirtqueue_add(&vqpci->vq, req_addr + 16, 512, true, true);

    qvirtqueue_add(&vqpci->vq, req_addr + 528, 1, true, false);



    qvirtqueue_kick(&qvirtio_pci, &dev->vdev, &vqpci->vq, free_head);





    qvirtio_wait_queue_isr(&qvirtio_pci, &dev->vdev, &vqpci->vq,

                           QVIRTIO_BLK_TIMEOUT_US);



    status = readb(req_addr + 528);

    g_assert_cmpint(status, ==, 0);



    data = g_malloc0(512);

    memread(req_addr + 16, data, 512);

    g_assert_cmpstr(data, ==, "TEST");

    g_free(data);



    guest_free(alloc, req_addr);



    /* End test */

    guest_free(alloc, vqpci->vq.desc);

    pc_alloc_uninit(alloc);

    qpci_msix_disable(dev->pdev);

    qvirtio_pci_device_disable(dev);

    g_free(dev);

    qpci_free_pc(bus);

    test_end();

}
