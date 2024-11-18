static void vfio_rtl8168_window_quirk_write(void *opaque, hwaddr addr,

                                            uint64_t data, unsigned size)

{

    VFIOQuirk *quirk = opaque;

    VFIOPCIDevice *vdev = quirk->vdev;



    switch (addr) {

    case 4: /* address */

        if ((data & 0x7fff0000) == 0x10000) {

            if (data & 0x10000000U &&

                vdev->pdev.cap_present & QEMU_PCI_CAP_MSIX) {



                trace_vfio_rtl8168_window_quirk_write_table(

                        memory_region_name(&quirk->mem),

                        vdev->vbasedev.name);



                memory_region_dispatch_write(&vdev->pdev.msix_table_mmio,

                                             (hwaddr)(quirk->data.address_match

                                                      & 0xfff),

                                             data,

                                             size,

                                             MEMTXATTRS_UNSPECIFIED);

            }



            quirk->data.flags = 1;

            quirk->data.address_match = data;



            return;

        }

        quirk->data.flags = 0;

        break;

    case 0: /* data */

        quirk->data.address_mask = data;

        break;

    }



    trace_vfio_rtl8168_window_quirk_write_direct(

            memory_region_name(&quirk->mem),

            vdev->vbasedev.name);



    vfio_region_write(&vdev->bars[quirk->data.bar].region,

                      addr + 0x70, data, size);

}
