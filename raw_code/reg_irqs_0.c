static int reg_irqs(CPUS390XState *env, S390PCIBusDevice *pbdev, ZpciFib fib)

{

    int ret, len;



    ret = css_register_io_adapter(S390_PCIPT_ADAPTER,

                                  FIB_DATA_ISC(ldl_p(&fib.data)), true, false,

                                  &pbdev->routes.adapter.adapter_id);

    assert(ret == 0);



    pbdev->summary_ind = get_indicator(ldq_p(&fib.aisb), sizeof(uint64_t));

    len = BITS_TO_LONGS(FIB_DATA_NOI(ldl_p(&fib.data))) * sizeof(unsigned long);

    pbdev->indicator = get_indicator(ldq_p(&fib.aibv), len);



    map_indicator(&pbdev->routes.adapter, pbdev->summary_ind);

    map_indicator(&pbdev->routes.adapter, pbdev->indicator);



    pbdev->routes.adapter.summary_addr = ldq_p(&fib.aisb);

    pbdev->routes.adapter.summary_offset = FIB_DATA_AISBO(ldl_p(&fib.data));

    pbdev->routes.adapter.ind_addr = ldq_p(&fib.aibv);

    pbdev->routes.adapter.ind_offset = FIB_DATA_AIBVO(ldl_p(&fib.data));

    pbdev->isc = FIB_DATA_ISC(ldl_p(&fib.data));

    pbdev->noi = FIB_DATA_NOI(ldl_p(&fib.data));

    pbdev->sum = FIB_DATA_SUM(ldl_p(&fib.data));



    DPRINTF("reg_irqs adapter id %d\n", pbdev->routes.adapter.adapter_id);

    return 0;

}
