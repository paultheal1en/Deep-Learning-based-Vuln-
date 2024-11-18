static void vmxnet3_adjust_by_guest_type(VMXNET3State *s)

{

    struct Vmxnet3_GOSInfo gos;



    VMXNET3_READ_DRV_SHARED(s->drv_shmem, devRead.misc.driverInfo.gos,

                            &gos, sizeof(gos));

    s->rx_packets_compound =

        (gos.gosType == VMXNET3_GOS_TYPE_WIN) ? false : true;



    VMW_CFPRN("Guest type specifics: RXCOMPOUND: %d", s->rx_packets_compound);

}
